import pygame
from config import VENTANA, ANCHO_VENTANA
from grid import crear_grid, dibujar, obtener_click_pos
from a_estrella import empezar_a_estrella

def main(ventana, ancho):
    FILAS = 11
    grid = crear_grid(FILAS, ancho)

    inicio = None
    fin = None

    corriendo = True

    while corriendo: 
        dibujar(ventana, grid, FILAS, ancho)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                corriendo = False

            # Resetear nodos con la tecla 'r'
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    inicio = None
                    fin = None
                    grid = crear_grid(FILAS, ancho)

            if pygame.mouse.get_pressed()[0]:  # Click izquierdo
                pos = pygame.mouse.get_pos()
                fila, col = obtener_click_pos(pos, FILAS, ancho)
                nodo = grid[fila][col]
                if not inicio and nodo != fin:
                    inicio = nodo
                    inicio.hacer_inicio()

                elif not fin and nodo != inicio:
                    fin = nodo
                    fin.hacer_fin()

                elif nodo != fin and nodo != inicio:
                    nodo.hacer_pared()

            elif pygame.mouse.get_pressed()[2]:  # Click derecho
                pos = pygame.mouse.get_pos()
                fila, col = obtener_click_pos(pos, FILAS, ancho)
                nodo = grid[fila][col]
                nodo.restablecer()
                if nodo == inicio:
                    inicio = None
                elif nodo == fin:
                    fin = None

            elif event.type == pygame.KEYDOWN: # Empezar el algoritmo A* pulsando ESPACIO
                if event.key == pygame.K_SPACE and inicio and fin:
                    empezar_a_estrella(inicio, fin, grid)

            

    pygame.quit()


if __name__ == "__main__":
    main(VENTANA, ANCHO_VENTANA)
