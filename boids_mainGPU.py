import glfw
from OpenGL.GL import *
import numpy as np
import math
import sys
import ctypes # <-- Importamos ctypes para la depuración

# --- 1. Constantes ---
WIDTH, HEIGHT = 1280, 720
NUM_BOIDS = 10000 
VIEW_RADIUS = 75.0
SEPARATION_DISTANCE = 25.0
MAX_SPEED = 3.0

# --- 2. Shaders (¡CORREGIDOS!) ---

# --- COMPUTE SHADER (CON ARREGLOS DE DIVISIÓN POR CERO) ---
COMPUTE_SHADER_SOURCE = """
#version 430 core

struct Boid {
    vec2 pos;
    vec2 vel;
};

layout (std430, binding = 0) buffer BoidsIn {
    Boid boids_in[];
};
layout (std430, binding = 1) buffer BoidsOut {
    Boid boids_out[];
};

uniform uint u_num_boids;
uniform float u_view_radius;
uniform float u_sep_dist;
uniform float u_max_speed;
uniform float u_width;
uniform float u_height;
uniform vec2 u_mouse_pos;

layout (local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

vec2 wrap_boundaries(vec2 pos) {
    if (pos.x < 0) { pos.x = u_width; }
    if (pos.x > u_width) { pos.x = 0; }
    if (pos.y < 0) { pos.y = u_height; }
    if (pos.y > u_height) { pos.y = 0; }
    return pos;
}

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= u_num_boids) { return; }

    vec2 pos_i = boids_in[i].pos;
    vec2 vel_i = boids_in[i].vel;

    vec2 sep_force = vec2(0.0);
    vec2 ali_force = vec2(0.0);
    vec2 coh_force = vec2(0.0);
    float count = 0.0;

    for (int j = 0; j < u_num_boids; j++) {
        if (i == j) continue;

        vec2 pos_j = boids_in[j].pos;
        vec2 vel_j = boids_in[j].vel;
        
        vec2 dist_vec = pos_i - pos_j;
        float dist = length(dist_vec);

        if (dist > 0.0 && dist < u_view_radius) {
            count += 1.0;
            
            if (dist < u_sep_dist) {
                sep_force += dist_vec / (dist * dist + 1e-6);
            }
            ali_force += vel_j;
            coh_force += pos_j;
        }
    }

    if (count > 0.0) {
        // --- Alineación (¡CORREGIDA!) ---
        ali_force /= count;
        float ali_len = length(ali_force);
        if (ali_len > 1e-6) { // Solo normalizar si el vector no es (0,0)
            ali_force = (ali_force / ali_len) * u_max_speed;
        }
        ali_force = ali_force - vel_i; // Steer
        
        // --- Cohesión (¡CORREGIDA!) ---
        coh_force /= count;
        vec2 vec_to_center = coh_force - pos_i;
        float coh_len = length(vec_to_center);
        if (coh_len > 1e-6) { // Solo normalizar si el vector no es (0,0)
            coh_force = (coh_force / coh_len) * u_max_speed;
        }
        coh_force = coh_force - vel_i; // Steer

        vel_i += (ali_force * 1.0) + (coh_force * 0.5) + (sep_force * 1.5);
    }
    
    // --- 4ª Regla: Evasión del Ratón ---
    float dist_to_mouse = distance(pos_i, u_mouse_pos);
    if (dist_to_mouse < 100.0) {
        vec2 repel_vec = pos_i - u_mouse_pos;
        vel_i += (repel_vec / (dist_to_mouse * dist_to_mouse + 1e-6)) * 5.0;
    }

    // --- Actualización Final (Corregida) ---
    float speed = length(vel_i);
    if (speed > u_max_speed) {
        vel_i = (vel_i / (speed + 1e-6)) * u_max_speed;
    }
    
    pos_i += vel_i;
    pos_i = wrap_boundaries(pos_i);
    
    boids_out[i].pos = pos_i;
    boids_out[i].vel = vel_i;
}
"""

# --- VERTEX SHADER (¡CORREGIDO!) ---
VERTEX_SHADER_SOURCE = """
#version 430 core

struct Boid {
    vec2 pos;
    vec2 vel;
};
layout (std430, binding = 0) buffer BoidsIn {
    Boid boids_in[]; // El array se llama 'boids_in'
};

uniform float u_width;
uniform float u_height;

void main() {
    // Leemos la posición del boid usando el nombre 'boids_in'
    vec2 pos_i = boids_in[gl_VertexID].pos;
    
    // Convertimos de (0, WIDTH) a (-1, 1) en espacio de clip
    vec2 pos_clip = (pos_i / vec2(u_width, u_height)) * 2.0 - 1.0;
    
    gl_Position = vec4(pos_clip, 0.0, 1.0);
}
"""

# --- FRAGMENT SHADER (Simple) ---
FRAGMENT_SHADER_SOURCE = """
#version 430 core
out vec4 FragColor;
void main() {
    FragColor = vec4(1.0, 1.0, 1.0, 1.0);
}
"""

# --- 3. Variables Globales de Python ---
compute_program = None
render_program = None
ssbo = [None, None]
vao = None
mouse_pos = (0.0, 0.0)
frame_index = 0

# --- 4. Funciones de Ayuda de OpenGL ---
def create_shader_program(compute_source=None, vertex_source=None, fragment_source=None):
    def compile_shader(source, shader_type):
        shader = glCreateShader(shader_type)
        glShaderSource(shader, source)
        glCompileShader(shader)
        if not glGetShaderiv(shader, GL_COMPILE_STATUS):
            log = glGetShaderInfoLog(shader).decode()
            raise Exception(f"Error compilando shader: {log}")
        return shader

    program = glCreateProgram()
    
    if compute_source:
        shader = compile_shader(compute_source, GL_COMPUTE_SHADER)
        glAttachShader(program, shader)
    if vertex_source:
        shader = compile_shader(vertex_source, GL_VERTEX_SHADER)
        glAttachShader(program, shader)
    if fragment_source:
        shader = compile_shader(fragment_source, GL_FRAGMENT_SHADER)
        glAttachShader(program, shader)

    glLinkProgram(program)
    if not glGetProgramiv(program, GL_LINK_STATUS):
        log = glGetProgramInfoLog(program).decode()
        raise Exception(f"Error enlazando programa: {log}")
        
    return program

def init_boids_buffers():
    global ssbo
    boid_data = np.zeros((NUM_BOIDS, 4), dtype=np.float32)
    for i in range(NUM_BOIDS):
        pos = [np.random.rand() * WIDTH, np.random.rand() * HEIGHT]
        angle = np.random.rand() * 2 * math.pi
        vel = [math.cos(angle), math.sin(angle)]
        boid_data[i, 0] = pos[0]
        boid_data[i, 1] = pos[1]
        boid_data[i, 2] = vel[0]
        boid_data[i, 3] = vel[1]

    ssbo = glGenBuffers(2)
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo[0])
    glBufferData(GL_SHADER_STORAGE_BUFFER, boid_data.nbytes, boid_data, GL_DYNAMIC_DRAW)
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo[1])
    glBufferData(GL_SHADER_STORAGE_BUFFER, boid_data.nbytes, None, GL_DYNAMIC_DRAW)
    print(f"Búferes SSBO creados para {NUM_BOIDS} boids.")

# --- 5. Funciones de Callback de GLFW ---
def on_mouse_move(window, xpos, ypos):
    global mouse_pos
    mouse_pos = (xpos, HEIGHT - ypos)

# --- 6. ¡FUNCIÓN DE DEPURACIÓN CORREGIDA! ---
def debug_read_ssbo_data(ssbo_index):
    """
    Lee los datos de un SSBO (GPU) y los trae a la CPU (NumPy) para imprimirlos.
    ¡ESTO ES MUY LENTO! Solo usar para depurar.
    """
    print(f"\n--- DEBUG: Leyendo SSBO {ssbo_index} ---")
    
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo[ssbo_index])
    size = glGetBufferParameteriv(GL_SHADER_STORAGE_BUFFER, GL_BUFFER_SIZE)
    
    if size != NUM_BOIDS * 4 * 4: # N * 4 floats * 4 bytes/float
        print(f"Error de depuración: Tamaño de SSBO incorrecto. Esperado: {NUM_BOIDS * 4 * 4}, Obtenido: {size}")
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)
        return

    data_ptr = glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_READ_ONLY)
    
    if not data_ptr:
        print("Error de depuración: glMapBuffer devolvió un puntero nulo.")
        glUnmapBuffer(GL_SHADER_STORAGE_BUFFER)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)
        return
        
    try:
        # --- ¡LECTURA ROBUSTA! ---
        # 1. Leer los bytes crudos desde el puntero de memoria
        buffer_data = ctypes.string_at(data_ptr, size)
        # 2. Convertir los bytes a un array de numpy
        data_copy = np.frombuffer(buffer_data, dtype=np.float32).reshape(NUM_BOIDS, 4)
        # --- FIN DE LA LECTURA ROBUSTA ---
    finally:
        glUnmapBuffer(GL_SHADER_STORAGE_BUFFER)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)
    
    # --- Análisis ---
    if np.isnan(data_copy).any():
        print("¡¡¡ERROR!!! Se han encontrado valores NaN en los datos de la GPU.")
        # Encontramos la primera fila (boid) que tiene un NaN
        nan_boid_index = np.where(np.isnan(data_copy).any(axis=1))[0][0]
        nan_boid = data_copy[nan_boid_index]
        print(f"  Ejemplo de Boid corrupto (Índice {nan_boid_index}): Pos={nan_boid[0:2]}, Vel={nan_boid[2:4]}")
    elif np.isinf(data_copy).any():
        print("¡¡¡ERROR!!! Se han encontrado valores Infinitos en los datos de la GPU.")
    else:
        print("Datos de la GPU parecen válidos (No hay NaN/Inf).")
        print("  Primeros 3 Boids:")
        print(data_copy[:3]) # Esto ahora SÍ imprimirá 3 filas
        
# --- 7. Función Principal (Con llamada de depuración) ---
def main():
    global compute_program, render_program, ssbo, vao, frame_index, mouse_pos
    
    if not glfw.init():
        print("Error: No se pudo inicializar GLFW.")
        sys.exit(-1)
        
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL_TRUE)

    window = glfw.create_window(WIDTH, HEIGHT, "Boids (100% GLSL Compute Shader)", None, None)
    if not window:
        glfw.terminate()
        print("Error: No se pudo crear la ventana de GLFW. ¿Tu GPU soporta OpenGL 4.3?")
        sys.exit(-1)

    glfw.make_context_current(window)
    print(f"Versión de OpenGL: {glGetString(GL_VERSION).decode('utf-8')}")
    glfw.swap_interval(1)
    
    try:
        compute_program = create_shader_program(compute_source=COMPUTE_SHADER_SOURCE)
        render_program = create_shader_program(vertex_source=VERTEX_SHADER_SOURCE, fragment_source=FRAGMENT_SHADER_SOURCE)
    except Exception as e:
        print(f"Error de Shader: {e}")
        glfw.terminate()
        sys.exit(-1)

    init_boids_buffers()
    vao = glGenVertexArrays(1)
    glfw.set_cursor_pos_callback(window, on_mouse_move)

    print("Iniciando... Cierra la ventana para salir. Mueve el ratón para repeler.")
    while not glfw.window_should_close(window):
        glfw.poll_events()

        glUseProgram(compute_program)
        
        glUniform1ui(glGetUniformLocation(compute_program, "u_num_boids"), NUM_BOIDS)
        glUniform1f(glGetUniformLocation(compute_program, "u_view_radius"), VIEW_RADIUS)
        glUniform1f(glGetUniformLocation(compute_program, "u_sep_dist"), SEPARATION_DISTANCE)
        glUniform1f(glGetUniformLocation(compute_program, "u_max_speed"), MAX_SPEED)
        glUniform1f(glGetUniformLocation(compute_program, "u_width"), WIDTH)
        glUniform1f(glGetUniformLocation(compute_program, "u_height"), HEIGHT)
        glUniform2f(glGetUniformLocation(compute_program, "u_mouse_pos"), mouse_pos[0], mouse_pos[1])
        
        input_buffer_index = frame_index % 2
        output_buffer_index = (frame_index + 1) % 2
        
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssbo[input_buffer_index])
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, ssbo[output_buffer_index])

        num_groups = (NUM_BOIDS + 255) // 256
        glDispatchCompute(num_groups, 1, 1)
        
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)
        
        # --- LLAMADA DE DEPURACIÓN (Solo en el primer fotograma) ---
        if frame_index == 0:
            debug_read_ssbo_data(output_buffer_index)

        glClearColor(0.1, 0.1, 0.15, 1.0)
        glClear(GL_COLOR_BUFFER_BIT)
        
        glUseProgram(render_program)
        
        glUniform1f(glGetUniformLocation(render_program, "u_width"), WIDTH)
        glUniform1f(glGetUniformLocation(render_program, "u_height"), HEIGHT)
        
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssbo[output_buffer_index])
        
        glPointSize(2.0)
        glBindVertexArray(vao)
        glDrawArrays(GL_POINTS, 0, NUM_BOIDS)
        glBindVertexArray(0)

        glfw.swap_buffers(window)
        frame_index += 1

    print("Cerrando aplicación...")
    glfw.terminate()

if __name__ == "__main__":
    main()