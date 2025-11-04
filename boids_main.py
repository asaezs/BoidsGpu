import glfw
from OpenGL.GL import *
import sys
import numpy as np
import math
from numba import cuda

USE_GPU = True

WIDTH, HEIGHT = 1280, 720

NUM_BOIDS = 1000
VIEW_RADIUS = 75.0
SEPARATION_DISTANCE = 25.0
MAX_SPEED = 3.0
MAX_FORCE = 0.1

# (N, 2) -> N boids, con 2 coords (x, y)
positions = np.zeros((NUM_BOIDS, 2), dtype=np.float32) # Float32 para OpenGL
velocities = np.zeros((NUM_BOIDS, 2), dtype=np.float32)

# Variables GPU
separation_force = np.zeros((NUM_BOIDS, 2), dtype=np.float32)
alignment_force = np.zeros((NUM_BOIDS, 2), dtype=np.float32)
cohesion_force = np.zeros((NUM_BOIDS, 2), dtype=np.float32)
neighbor_count = np.zeros((NUM_BOIDS, 1), dtype=np.float32)

def init_boids():
    global positions, velocities
    for i in range(NUM_BOIDS):
        # Posición inicial aleatoria
        positions[i] = [np.random.rand() * WIDTH, np.random.rand() * HEIGHT]
        # Velocidad aleatoria y normalizada
        angle = np.random.rand() * 2 * math.pi
        velocities[i] = [math.cos(angle), math.sin(angle)]

# Función que usa la CPU para simular los BOIDS (LENTO)
def update_boids_cpu():
    global positions, velocities
    
    # Creamos una copia para no interferir con los cálculos
    new_velocities = np.copy(velocities)
    
    for i in range(NUM_BOIDS):
        # 3 reglas de los boids
        separation = np.zeros(2, dtype=np.float32)
        alignment = np.zeros(2, dtype=np.float32)
        cohesion = np.zeros(2, dtype=np.float32)
        
        neighbor_count = 0
        
        # O(N^2) Cada boid comprueba a todos los demás
        for j in range(NUM_BOIDS):
            if i == j:
                continue
                
            # Calcular la distancia
            dist_vec = positions[i] - positions[j]
            distance = math.sqrt(dist_vec[0]**2 + dist_vec[1]**2)
            
            # Si entra en rango
            if distance > 0 and distance < VIEW_RADIUS:
                neighbor_count += 1
                
                # Separación
                if distance < SEPARATION_DISTANCE:
                    # Añadir una fuerza de repulsión
                    separation += dist_vec / (distance * distance) # A más cerca más rápido
                
                # Alineación
                alignment += velocities[j]
                
                # Cohesión
                cohesion += positions[j]

        if neighbor_count > 0:
            # Promedio de las fuerzas de vecinos
            alignment /= neighbor_count
            alignment = (alignment / (np.linalg.norm(alignment) + 1e-6)) * MAX_SPEED
            steer_alignment = alignment - velocities[i]
            
            cohesion /= neighbor_count
            vec_to_center = cohesion - positions[i] # Vector hacia el centro de masa
            cohesion = (vec_to_center / (np.linalg.norm(vec_to_center) + 1e-6)) * MAX_SPEED
            steer_cohesion = cohesion - velocities[i]
            
            # Aplicar las fuerzas
            new_velocities[i] += (steer_alignment * 1.0)
            new_velocities[i] += (steer_cohesion * 0.5)
            new_velocities[i] += (separation * 1.5)

    # Actualizar
    velocities = new_velocities
    
    for i in range(NUM_BOIDS):
        # Limitar velocidad
        speed = np.linalg.norm(velocities[i])
        if speed > MAX_SPEED:
            velocities[i] = (velocities[i] / speed) * MAX_SPEED
            
        # Mover boid
        positions[i] += velocities[i]
        
        # Wrapping
        wrap_boundaries(i)

# Como el for de update_boids_cpu pero usando la GPU
@cuda.jit
def boids_rules_kernel(positions_in, velocities_in, separation_out, alignment_out, cohesion_out, neighbor_count_out,
                       view_radius, separation_distance):
    """ Cada hilo de la GPU es responsable de UN solo boid (índice 'i'). """
    
    # Esto es como la iteracion del bucle for de la cpu
    i = cuda.grid(1)
    
    # Dentro de los límites
    if i >= positions_in.shape[0]:
        return

    s_force_x, s_force_y = 0.0, 0.0
    a_force_x, a_force_y = 0.0, 0.0
    c_force_x, c_force_y = 0.0, 0.0
    count = 0

    # Bucle para cada hilo de la GPU (Boid)
    for j in range(positions_in.shape[0]):
        if i == j:
            continue
            
        # Calcular la distancia
        dist_x = positions_in[i, 0] - positions_in[j, 0]
        dist_y = positions_in[i, 1] - positions_in[j, 1]
        distance = math.sqrt(dist_x**2 + dist_y**2)

        # Si entra en rango
        if distance > 0 and distance < view_radius:
            count += 1
            
            # Separación
            if distance < separation_distance:
                s_force_x += dist_x / (distance * distance + 1e-6)
                s_force_y += dist_y / (distance * distance + 1e-6)

            # Alineación
            a_force_x += velocities_in[j, 0]
            a_force_y += velocities_in[j, 1]

            # Cohesión
            c_force_x += positions_in[j, 0]
            c_force_y += positions_in[j, 1]

    # Escribir los resultados en la memoria de salida de la GPU
    separation_out[i, 0] = s_force_x
    separation_out[i, 1] = s_force_y
    alignment_out[i, 0] = a_force_x
    alignment_out[i, 1] = a_force_y
    cohesion_out[i, 0] = c_force_x
    cohesion_out[i, 1] = c_force_y
    neighbor_count_out[i, 0] = count

def update_boids_gpu():
    global positions, velocities, separation_force, alignment_force, cohesion_force, neighbor_count
    
    # 1 hilo por boid
    threads_per_block = 256
    blocks_per_grid = (NUM_BOIDS + threads_per_block - 1) // threads_per_block

    # De RAM a VRAM
    d_positions_in = cuda.to_device(positions)
    d_velocities_in = cuda.to_device(velocities)
    
    # Crear arrays de salida vacíos en GPU
    d_separation_out = cuda.device_array_like(separation_force)
    d_alignment_out = cuda.device_array_like(alignment_force)
    d_cohesion_out = cuda.device_array_like(cohesion_force)
    d_neighbor_count_out = cuda.device_array_like(neighbor_count)

    # La GPU ahora está haciendo el trabajo O(N^2)
    boids_rules_kernel[blocks_per_grid, threads_per_block](
        d_positions_in, d_velocities_in,
        d_separation_out, d_alignment_out, d_cohesion_out, d_neighbor_count_out,
        VIEW_RADIUS, SEPARATION_DISTANCE
    )
    
    # Esperar a que la GPU termine
    cuda.synchronize()

    # VRAM a RAM
    separation_force = d_separation_out.copy_to_host()
    alignment_force = d_alignment_out.copy_to_host()
    cohesion_force = d_cohesion_out.copy_to_host()
    neighbor_count = d_neighbor_count_out.copy_to_host()

    # Vuelta a la CPU (Reutilizamos el código de la CPU para aplicar las fuerzas)
    
    # Evitar división por cero
    for i in range(NUM_BOIDS):
        if neighbor_count[i, 0] > 0:
            count = neighbor_count[i, 0]
            
            alignment_force[i] /= count
            norm = math.sqrt(alignment_force[i, 0]**2 + alignment_force[i, 1]**2)
            alignment_force[i] = (alignment_force[i] / (norm + 1e-6)) * MAX_SPEED
            steer_alignment = alignment_force[i] - velocities[i]

            cohesion_force[i] /= count
            vec_to_center = cohesion_force[i] - positions[i]
            norm = math.sqrt(vec_to_center[0]**2 + vec_to_center[1]**2)
            cohesion_force[i] = (vec_to_center / (norm + 1e-6)) * MAX_SPEED
            steer_cohesion = cohesion_force[i] - velocities[i]

            # Aplicar Fuerzas
            velocities[i] += (steer_alignment * 1.0)
            velocities[i] += (steer_cohesion * 0.5)
            velocities[i] += (separation_force[i] * 1.5)

    # Actualizar
    for i in range(NUM_BOIDS):
        speed = math.sqrt(velocities[i, 0]**2 + velocities[i, 1]**2)
        if speed > MAX_SPEED:
            velocities[i] = (velocities[i] / speed) * MAX_SPEED
            
        positions[i] += velocities[i]
        wrap_boundaries(i)

def wrap_boundaries(i):
    if positions[i, 0] < 0: positions[i, 0] = WIDTH
    if positions[i, 0] > WIDTH: positions[i, 0] = 0
    if positions[i, 1] < 0: positions[i, 1] = HEIGHT
    if positions[i, 1] > HEIGHT: positions[i, 1] = 0

def draw_boids():
    glPointSize(2.0)
    glColor3f(1.0, 1.0, 1.0)
    
    # Algo lento pero sencillo
    glBegin(GL_POINTS)
    for i in range(NUM_BOIDS):
        glVertex2f(positions[i, 0], positions[i, 1])
    glEnd()

def main():
    
    if not glfw.init():
        print("Error: No se pudo inicializar GLFW.")
        sys.exit(-1)

    # Crear ventana
    window = glfw.create_window(WIDTH, HEIGHT, "Boids", None, None)
    if not window:
        glfw.terminate()
        print("Error: No se pudo crear la ventana de GLFW.")
        sys.exit(-1)

    # Indicar donde operar a OpenGL
    glfw.make_context_current(window)
    
    print(f"Versión de OpenGL: {glGetString(GL_VERSION).decode('utf-8')}")

    # V-Sync
    glfw.swap_interval(1)

    # Le decimos a OpenGL que queremos una cámara 2D que
    # coincida exactamente con nuestras coordenadas de píxeles
    glViewport(0, 0, WIDTH, HEIGHT)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    glOrtho(0, WIDTH, 0, HEIGHT, -1, 1) # Coords de 0-WIDTH en X, 0-HEIGHT en Y
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

    init_boids()

    if USE_GPU and cuda.is_available():
        print("¡GPU detectada! Usando el kernel de CUDA.")
        update_function = update_boids_gpu
        glfw.set_window_title(window, "Simulador de Boids GPU")
    else:
        print("GPU no detectada o desactivada. Usando la CPU (lento).")
        update_function = update_boids_cpu
        glfw.set_window_title(window, "Simulador de Boids CPU")

    print("Iniciando... Cierra la ventana para salir.")
    while not glfw.window_should_close(window):
        
        glfw.poll_events()

        update_function()
        
        glClearColor(0.1, 0.1, 0.15, 1.0)
        
        glClear(GL_COLOR_BUFFER_BIT)

        draw_boids()

        # Del Back Buffer (OpenGL) al Front Buffer (Pantalla)
        # Esto hace que OpenGL calcule por debajo y a posteriori
        # muestre lo que ha calculado, para evitar ver calculos a medias
        glfw.swap_buffers(window)

    print("Cerrando aplicación...")
    glfw.terminate()

if __name__ == "__main__":
    main()