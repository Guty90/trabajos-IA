import pygame
from config import BLANCO, NEGRO, NARANJA, PURPURA, GRIS_CLARO, TURQUESA, AMARILLO, ROJO_CLARO

class Nodo:
    def __init__(self, fila, col, ancho, total_filas):
        self.fila = fila
        self.col = col
        self.x = fila * ancho
        self.y = col * ancho
        self.color = BLANCO
        self.ancho = ancho
        self.total_filas = total_filas
        self.g = 0
        self.h = 0
        self.f = 0
        self.padre = None

    def get_vecinos(self, grid):
        vecinos = []
        for i in range(max(0, self.fila - 1), min(self.total_filas, self.fila + 2)):
            for j in range(max(0, self.col - 1), min(self.total_filas, self.col + 2)):
                if (i != self.fila or j != self.col) and not grid[i][j].es_pared():
                    vecinos.append(grid[i][j])
        return vecinos

    def get_pos(self):
        return self.fila, self.col

    def es_pared(self):
        return self.color == NEGRO

    def es_inicio(self):
        return self.color == NARANJA

    def es_fin(self):
        return self.color == PURPURA

    def es_camino(self):
        return self.color == TURQUESA

    def restablecer(self):
        self.color = BLANCO

    def hacer_inicio(self):
        self.color = NARANJA

    def hacer_pared(self):
        self.color = NEGRO

    def hacer_fin(self):
        self.color = PURPURA

    def hacer_camino(self):
        self.color = TURQUESA

    def hacer_no_es_camino(self):
        self.color = GRIS_CLARO

    def hacer_abierto(self):
        self.color = AMARILLO
    
    def hacer_cerrado(self):
        self.color = ROJO_CLARO

    def dibujar(self, ventana):
        pygame.draw.rect(ventana, self.color, (self.x, self.y, self.ancho, self.ancho))
