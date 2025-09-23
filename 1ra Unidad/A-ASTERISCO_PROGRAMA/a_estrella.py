import pygame
from nodo import Nodo

def obtener_distancia_manhattan(nodo1, nodo2):
    # Calcula la distancia Manhattan entre dos nodos
    x1, y1 = nodo1.get_pos()
    x2, y2 = nodo2.get_pos()
    return abs(x1 - x2) + abs(y1 - y2)

def reconstruir_camino(nodo):
    camino = []
    while nodo:
        camino.append(nodo)
        nodo = nodo.padre
    camino.reverse()
    for n in camino:
        if not n.es_inicio() and not n.es_fin():
            n.hacer_camino()
    return camino

def pintar_abiertos(lista_abierta):
    for nodo in lista_abierta:
        if not nodo.es_inicio() and not nodo.es_fin() and not nodo.es_camino():
            nodo.hacer_abierto()

def pintar_cerrados(lista_cerrada):
    for nodo in lista_cerrada:
        if not nodo.es_inicio() and not nodo.es_fin() and not nodo.es_camino():
            nodo.hacer_cerrado()

def empezar_a_estrella(inicio, fin, grid):
    # Aquí iría la implementación del algoritmo A*
    lista_abierta = [inicio] # nodos por explorar
    lista_cerrada = set() # nodos ya explorados

    while lista_abierta:
        nodo_actual = min(lista_abierta, key=lambda n: n.f) # nodo con menor costo total
        lista_abierta.remove(nodo_actual)
        lista_cerrada.add(nodo_actual)

        if nodo_actual == fin:
            # pintar_abiertos(lista_abierta)
            # pintar_cerrados(lista_cerrada)
            return reconstruir_camino(nodo_actual)

        for vecino in nodo_actual.get_vecinos(grid):
            if vecino in lista_cerrada:
                continue

            # Calcular costos
            nuevo_g = nodo_actual.g + 1
            if vecino not in lista_abierta:
                lista_abierta.append(vecino)
            elif nuevo_g >= vecino.g:
                continue

            # Actualizar costos
            vecino.g = nuevo_g
            vecino.h = obtener_distancia_manhattan(vecino, fin)
            vecino.f = vecino.g + vecino.h
            vecino.padre = nodo_actual

        
        
