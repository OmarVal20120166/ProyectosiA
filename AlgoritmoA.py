import pygame
from queue import PriorityQueue


ANCHO_VENTANA = 800
VENTANA = pygame.display.set_mode((ANCHO_VENTANA, ANCHO_VENTANA))
pygame.display.set_caption("Visualización de Nodos con A*")


BLANCO = (255, 255, 255)
NEGRO = (0, 0, 0)
GRIS = (128, 128, 128)
VERDE = (0, 255, 0)
ROJO = (255, 0, 0)
NARANJA = (255, 165, 0)
PURPURA = (128, 0, 128)
AZUL = (0, 0, 255)
AMARILLO = (255, 255, 0)

# Costos de movimiento
COSTO_CARDINAL = 10  # Arriba, abajo, izquierda, derecha
COSTO_DIAGONAL = 14  # Movimiento diagonal

class Nodo:
    def __init__(self, fila, col, ancho, total_filas):
        self.fila = fila
        self.col = col
        self.x = fila * ancho
        self.y = col * ancho
        self.color = BLANCO
        self.ancho = ancho
        self.total_filas = total_filas
        self.vecinos = []

    def get_pos(self):
        return self.fila, self.col

    def es_pared(self):
        return self.color == NEGRO

    def es_inicio(self):
        return self.color == NARANJA

    def es_fin(self):
        return self.color == PURPURA

    def restablecer(self):
        self.color = BLANCO

    def hacer_inicio(self):
        self.color = NARANJA

    def hacer_pared(self):
        self.color = NEGRO

    def hacer_fin(self):
        self.color = PURPURA

    def hacer_abierto(self):
        self.color = VERDE

    def hacer_cerrado(self):
        self.color = ROJO

    def hacer_camino(self):
        self.color = AZUL

    def dibujar(self, ventana):
        pygame.draw.rect(ventana, self.color, (self.x, self.y, self.ancho, self.ancho))

    # FUNCIÓN con diagonales
    def actualizar_vecinos(self, grid):
        self.vecinos = []
        # Movimientos cardinales
        if self.fila < self.total_filas - 1 and not grid[self.fila + 1][self.col].es_pared():  # Abajo
            self.vecinos.append((grid[self.fila + 1][self.col], COSTO_CARDINAL))
        if self.fila > 0 and not grid[self.fila - 1][self.col].es_pared():  # Arriba
            self.vecinos.append((grid[self.fila - 1][self.col], COSTO_CARDINAL))
        if self.col < self.total_filas - 1 and not grid[self.fila][self.col + 1].es_pared():  # Derecha
            self.vecinos.append((grid[self.fila][self.col + 1], COSTO_CARDINAL))
        if self.col > 0 and not grid[self.fila][self.col - 1].es_pared():  # Izquierda
            self.vecinos.append((grid[self.fila][self.col - 1], COSTO_CARDINAL))

        # Movimientos diagonales
        if self.fila > 0 and self.col > 0 and not grid[self.fila - 1][self.col - 1].es_pared():  # Arriba-Izq
            self.vecinos.append((grid[self.fila - 1][self.col - 1], COSTO_DIAGONAL))
        if self.fila > 0 and self.col < self.total_filas - 1 and not grid[self.fila - 1][self.col + 1].es_pared():  # Arriba-Der
            self.vecinos.append((grid[self.fila - 1][self.col + 1], COSTO_DIAGONAL))
        if self.fila < self.total_filas - 1 and self.col > 0 and not grid[self.fila + 1][self.col - 1].es_pared():  # Abajo-Izq
            self.vecinos.append((grid[self.fila + 1][self.col - 1], COSTO_DIAGONAL))
        if self.fila < self.total_filas - 1 and self.col < self.total_filas - 1 and not grid[self.fila + 1][self.col + 1].es_pared():  # Abajo-Der
            self.vecinos.append((grid[self.fila + 1][self.col + 1], COSTO_DIAGONAL))


def crear_grid(filas, ancho):
    grid = []
    ancho_nodo = ancho // filas
    for i in range(filas):
        grid.append([])
        for j in range(filas):
            nodo = Nodo(i, j, ancho_nodo, filas)
            grid[i].append(nodo)
    return grid

def dibujar_grid(ventana, filas, ancho):
    ancho_nodo = ancho // filas
    for i in range(filas):
        pygame.draw.line(ventana, GRIS, (0, i * ancho_nodo), (ancho, i * ancho_nodo))
        for j in range(filas):
            pygame.draw.line(ventana, GRIS, (j * ancho_nodo, 0), (j * ancho_nodo, ancho))

def dibujar(ventana, grid, filas, ancho):
    ventana.fill(BLANCO)
    for fila in grid:
        for nodo in fila:
            nodo.dibujar(ventana)
    dibujar_grid(ventana, filas, ancho)
    pygame.display.update()

def obtener_click_pos(pos, filas, ancho):
    ancho_nodo = ancho // filas
    y, x = pos
    fila = y // ancho_nodo
    col = x // ancho_nodo
    return fila, col


def heuristica(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    dx = abs(x1 - x2)
    dy = abs(y1 - y2)
    return COSTO_CARDINAL * (dx + dy)

def formatear_lista_nodos(nodos):
    """Devuelve las posiciones ordenadas de un conjunto de nodos para el log."""
    return [nodo.get_pos() for nodo in sorted(nodos, key=lambda n: (n.fila, n.col))]

def reconstruir_camino(came_from, actual, dibujar):
    while actual in came_from:
        actual = came_from[actual]
        actual.hacer_camino()
        dibujar()

def reiniciar_tablero(grid):
    """Restablece todos los nodos del grid a su estado inicial."""
    for fila in grid:
        for nodo in fila:
            nodo.restablecer()

def algoritmo_A_star(dibujar, grid, inicio, fin):
    count = 0
    open_set = PriorityQueue()
    open_set.put((0, count, inicio))
    came_from = {}

    g_score = {nodo: float("inf") for fila in grid for nodo in fila}
    f_score = {nodo: float("inf") for fila in grid for nodo in fila}

    g_score[inicio] = 0
    f_score[inicio] = heuristica(inicio.get_pos(), fin.get_pos())

    open_set_hash = {inicio}
    closed_set = set()

    print(f"\n Iniciando A* desde {inicio.get_pos()} hasta {fin.get_pos()}")

    while not open_set.empty():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        actual = open_set.get()[2]
        open_set_hash.remove(actual)

        print(f"\n Nodo actual: {actual.get_pos()} | g={g_score[actual]}, h={heuristica(actual.get_pos(), fin.get_pos())}, f={f_score[actual]}")
        print(f"    Abiertos: {formatear_lista_nodos(open_set_hash)}")
        print(f"    Cerrados: {formatear_lista_nodos(closed_set)}")

        if actual == fin:
            print("\n Camino encontrado.")
            reconstruir_camino(came_from, fin, dibujar)
            fin.hacer_fin()
            inicio.hacer_inicio()
            return True

        for vecino, costo in actual.vecinos:
            temp_g_score = g_score[actual] + costo
            h = heuristica(vecino.get_pos(), fin.get_pos())
            f = temp_g_score + h

            print(f"   Vecino {vecino.get_pos()}: g_actual={g_score.get(vecino, float('inf'))}, g_nuevo={temp_g_score}, h={h}, f={f}")

            if temp_g_score < g_score[vecino]:
                came_from[vecino] = actual
                g_score[vecino] = temp_g_score
                f_score[vecino] = f

                if vecino not in open_set_hash:
                    count += 1
                    open_set.put((f_score[vecino], count, vecino))
                    open_set_hash.add(vecino)
                    vecino.hacer_abierto()
                    print(f"      Vecino {vecino.get_pos()} agregado a la lista abierta")
            else:
                print(f"      Vecino {vecino.get_pos()} descartado (g_nuevo >= g_actual)")

        dibujar()

        if actual != inicio:
            actual.hacer_cerrado()
            closed_set.add(actual)
            print(f"    Nodo {actual.get_pos()} marcado como cerrado")
            print(f"    Estado actual -> Abiertos: {formatear_lista_nodos(open_set_hash)} | Cerrados: {formatear_lista_nodos(closed_set)}")

    print("\n No se encontró camino posible.")
    return False



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

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE and inicio and fin:
                    for fila in grid:
                        for nodo in fila:
                            nodo.actualizar_vecinos(grid)
                    algoritmo_A_star(lambda: dibujar(ventana, grid, FILAS, ancho), grid, inicio, fin)
                elif event.key == pygame.K_r:
                    reiniciar_tablero(grid)
                    inicio = None
                    fin = None

    pygame.quit()

main(VENTANA, ANCHO_VENTANA)
