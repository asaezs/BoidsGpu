import glfw
from OpenGL.GL import *
import sys
import numpy as np
import math

WIDTH, HEIGHT = 1280, 720

NUM_BOIDS = 1000
VIEW_RADIUS = 75.0
SEPARATION_DISTANCE = 25.0
MAX_SPEED = 3.0
MAX_FORCE = 0.1

# (N, 2) -> N boids, con 2 coords (x, y)
positions = np.zeros((NUM_BOIDS, 2), dtype=np.float32) # Float32 para OpenGL
velocities = np.zeros((NUM_BOIDS, 2), dtype=np.float32)

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

    print("Iniciando... Cierra la ventana para salir.")
    while not glfw.window_should_close(window):
        
        glfw.poll_events()

        update_boids_cpu()
        
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