import json
import random

# Dataset de instrucciones y respuestas sobre programación
dataset = []

# ========== NIVEL BÁSICO ==========

# Conceptos fundamentales
basics = [
    {
        "instruction": "¿Qué es una variable en programación?",
        "response": "Una variable es un espacio en memoria que almacena un valor que puede cambiar durante la ejecución del programa. Se le asigna un nombre identificador para poder acceder a ese valor. Por ejemplo, en Python: x = 5 almacena el número 5 en la variable x."
    },
    {
        "instruction": "Explica qué son los tipos de datos",
        "response": "Los tipos de datos definen qué clase de información puede almacenar una variable. Los tipos más comunes son: enteros (int), decimales (float), texto (string), booleanos (bool), y estructuras como listas y diccionarios. Cada tipo tiene operaciones específicas que se pueden realizar con él."
    },
    {
        "instruction": "¿Qué es un operador en programación?",
        "response": "Un operador es un símbolo que realiza una operación sobre uno o más valores. Existen operadores aritméticos (+, -, *, /), de comparación (==, !=, <, >), lógicos (and, or, not) y de asignación (=, +=, -=). Por ejemplo, 5 + 3 usa el operador aritmético de suma."
    },
    {
        "instruction": "¿Qué es un condicional if?",
        "response": "Un condicional if es una estructura de control que ejecuta código solo si se cumple una condición. Sintaxis básica: if condicion: ejecutar_esto. Puede combinarse con elif (else if) para múltiples condiciones y else para un caso por defecto. Ejemplo: if edad >= 18: print('Mayor de edad')"
    },
    {
        "instruction": "Explica qué es un bucle for",
        "response": "Un bucle for es una estructura que repite un bloque de código un número determinado de veces, iterando sobre una secuencia. En Python: for i in range(5): print(i) imprime números del 0 al 4. Es útil cuando sabes cuántas veces necesitas repetir algo."
    },
    {
        "instruction": "¿Qué es un bucle while?",
        "response": "Un bucle while repite código mientras una condición sea verdadera. Sintaxis: while condicion: ejecutar_esto. Es útil cuando no sabes cuántas iteraciones necesitas. Ejemplo: while x < 10: x += 1. Importante: asegurarse de que la condición eventualmente sea falsa para evitar bucles infinitos."
    },
    {
        "instruction": "¿Qué es una función?",
        "response": "Una función es un bloque de código reutilizable que realiza una tarea específica. Se define con def nombre(parametros): y puede retornar valores con return. Ejemplo: def suma(a, b): return a + b. Las funciones ayudan a organizar el código y evitar repetición."
    },
    {
        "instruction": "¿Qué son los parámetros de una función?",
        "response": "Los parámetros son variables que recibe una función para trabajar con ellos. Se declaran en la definición de la función entre paréntesis. Pueden tener valores por defecto. Ejemplo: def saludar(nombre, saludo='Hola'): return f'{saludo}, {nombre}'. Aquí 'nombre' es obligatorio y 'saludo' tiene un valor por defecto."
    },
    {
        "instruction": "Explica qué es una lista",
        "response": "Una lista es una estructura de datos que almacena múltiples elementos en orden. En Python se crean con corchetes: lista = [1, 2, 3]. Las listas son mutables (se pueden modificar), permiten duplicados, y se accede a sus elementos por índice: lista[0] devuelve el primer elemento."
    },
    {
        "instruction": "¿Qué es un diccionario?",
        "response": "Un diccionario es una estructura que almacena pares clave-valor. Se crea con llaves: dict = {'nombre': 'Juan', 'edad': 25}. Permiten acceso rápido a valores mediante sus claves: dict['nombre'] devuelve 'Juan'. Son ideales para representar objetos con propiedades."
    },
    {
        "instruction": "¿Qué es un string?",
        "response": "Un string (cadena de texto) es una secuencia de caracteres. Se define entre comillas simples o dobles: 'Hola' o \"Hola\". Son inmutables, pero se pueden concatenar con +, multiplicar con *, y tienen métodos útiles como .upper(), .lower(), .split(), .replace()."
    },
    {
        "instruction": "Explica qué es el índice en una lista",
        "response": "El índice es la posición numérica de un elemento en una lista, comenzando desde 0. lista[0] accede al primer elemento, lista[1] al segundo, etc. Los índices negativos cuentan desde el final: lista[-1] es el último elemento. Acceder a un índice fuera de rango causa un error."
    },
    {
        "instruction": "¿Qué es la concatenación?",
        "response": "La concatenación es unir dos o más strings en uno solo. En Python se usa el operador +: 'Hola' + ' ' + 'Mundo' resulta en 'Hola Mundo'. También se puede usar con listas para combinarlas: [1, 2] + [3, 4] resulta en [1, 2, 3, 4]."
    },
    {
        "instruction": "¿Qué son los comentarios en código?",
        "response": "Los comentarios son texto que el intérprete ignora, usado para documentar código. En Python se usan con #: # Este es un comentario. Para comentarios multilínea se usan triple comillas: '''Comentario largo'''. Son esenciales para explicar el propósito del código."
    },
    {
        "instruction": "Explica qué es la indentación",
        "response": "La indentación es el espaciado al inicio de una línea de código. En Python es fundamental ya que define bloques de código (no se usan llaves como en otros lenguajes). Generalmente se usan 4 espacios. Un error de indentación causa IndentationError. Ejemplo: if True:\\n    print('indentado')"
    }
]

dataset.extend(basics)

# ========== NIVEL INTERMEDIO ==========

intermediate = [
    {
        "instruction": "¿Qué es la recursión?",
        "response": "La recursión es cuando una función se llama a sí misma para resolver un problema dividiéndolo en subproblemas más pequeños. Necesita un caso base para terminar. Ejemplo: def factorial(n): return 1 if n == 0 else n * factorial(n-1). Es útil para problemas como recorrer árboles o calcular secuencias."
    },
    {
        "instruction": "Explica qué es la complejidad temporal",
        "response": "La complejidad temporal (notación Big O) mide cómo crece el tiempo de ejecución de un algoritmo con el tamaño de entrada. O(1) es constante, O(n) lineal, O(n²) cuadrática, O(log n) logarítmica. Por ejemplo, buscar en una lista desordenada es O(n), en una lista ordenada con búsqueda binaria es O(log n)."
    },
    {
        "instruction": "¿Qué es una clase en POO?",
        "response": "Una clase es un molde para crear objetos que define sus atributos y métodos. Ejemplo: class Persona: def __init__(self, nombre): self.nombre = nombre. Se instancia con persona = Persona('Ana'). Las clases implementan encapsulación, herencia y polimorfismo, pilares de la Programación Orientada a Objetos."
    },
    {
        "instruction": "¿Qué es el método __init__?",
        "response": "El método __init__ es el constructor de una clase en Python, se ejecuta automáticamente al crear un objeto. Define el estado inicial del objeto. Ejemplo: def __init__(self, nombre, edad): self.nombre = nombre; self.edad = edad. El parámetro self representa la instancia del objeto."
    },
    {
        "instruction": "Explica qué es la herencia",
        "response": "La herencia permite crear una clase que adquiere atributos y métodos de otra clase padre. Sintaxis: class Hijo(Padre). Ejemplo: class Estudiante(Persona): permite que Estudiante tenga todo de Persona más sus propios atributos. Usa super() para acceder a métodos del padre. Promueve reutilización de código."
    },
    {
        "instruction": "¿Qué es el polimorfismo?",
        "response": "El polimorfismo permite que objetos de diferentes clases respondan al mismo método de manera distinta. Ejemplo: tanto Perro como Gato pueden tener un método hablar(), pero uno retorna 'Guau' y otro 'Miau'. Esto permite escribir código más flexible y extensible."
    },
    {
        "instruction": "¿Qué es una excepción?",
        "response": "Una excepción es un error que ocurre durante la ejecución del programa. Se manejan con try-except: try: codigo_que_puede_fallar except TipoError: manejar_error. Esto evita que el programa se detenga abruptamente. Puedes lanzar excepciones con raise y crear excepciones personalizadas."
    },
    {
        "instruction": "Explica qué es una lambda",
        "response": "Una función lambda es una función anónima de una sola línea. Sintaxis: lambda parametros: expresion. Ejemplo: cuadrado = lambda x: x**2. Son útiles para funciones simples usadas como argumentos en map(), filter(), sorted(). No pueden contener múltiples expresiones ni statements."
    },
    {
        "instruction": "¿Qué es map() en Python?",
        "response": "map() aplica una función a cada elemento de un iterable. Sintaxis: map(funcion, iterable). Retorna un iterador. Ejemplo: list(map(lambda x: x*2, [1,2,3])) resulta en [2,4,6]. Es útil para transformar colecciones sin bucles explícitos."
    },
    {
        "instruction": "¿Qué es filter() en Python?",
        "response": "filter() filtra elementos de un iterable según una condición. Sintaxis: filter(funcion_bool, iterable). Ejemplo: list(filter(lambda x: x > 5, [3,6,2,8])) resulta en [6,8]. Retorna solo elementos donde la función retorna True."
    },
    {
        "instruction": "Explica qué son las list comprehensions",
        "response": "Las list comprehensions son una forma concisa de crear listas. Sintaxis: [expresion for item in iterable if condicion]. Ejemplo: [x**2 for x in range(10) if x % 2 == 0] crea una lista de cuadrados de números pares. Son más legibles y eficientes que bucles tradicionales."
    },
    {
        "instruction": "¿Qué es un generador en Python?",
        "response": "Un generador es una función que usa yield en lugar de return para producir una secuencia de valores de manera eficiente. Genera valores bajo demanda sin almacenarlos todos en memoria. Ejemplo: def contador(): n = 0; while True: yield n; n += 1. Se itera con next() o en bucles for."
    },
    {
        "instruction": "¿Qué es un decorador?",
        "response": "Un decorador es una función que modifica el comportamiento de otra función. Se aplica con @nombre_decorador encima de la función. Ejemplo: @timer puede medir el tiempo de ejecución. Sintaxis: def decorador(func): def wrapper(*args): # código antes; resultado = func(*args); # código después; return resultado; return wrapper."
    },
    {
        "instruction": "Explica qué es el scope de variables",
        "response": "El scope define dónde una variable es accesible. Existen scope local (dentro de funciones), global (nivel de módulo), y nonlocal (funciones anidadas). Las variables locales no son visibles fuera de su función. Para modificar una variable global desde una función se usa la palabra clave global."
    },
    {
        "instruction": "¿Qué es un módulo en Python?",
        "response": "Un módulo es un archivo .py que contiene código Python reutilizable. Se importa con import nombre_modulo o from modulo import funcion. Permite organizar código en archivos separados. Python tiene módulos estándar (math, random) y se pueden crear módulos personalizados."
    }
]

dataset.extend(intermediate)

# ========== ESTRUCTURAS DE DATOS ==========

data_structures = [
    {
        "instruction": "¿Qué es una tupla?",
        "response": "Una tupla es similar a una lista pero inmutable (no se puede modificar después de crearla). Se define con paréntesis: t = (1, 2, 3). Son más rápidas que las listas y se usan para datos que no deben cambiar, como coordenadas. Se accede por índice igual que las listas."
    },
    {
        "instruction": "¿Qué es un set?",
        "response": "Un set es una colección desordenada de elementos únicos (sin duplicados). Se crea con llaves: s = {1, 2, 3} o set([1,2,2]) que resulta en {1,2}. Soporta operaciones de conjuntos: unión, intersección, diferencia. Es muy eficiente para verificar pertenencia (in) con O(1)."
    },
    {
        "instruction": "Explica qué es una cola (queue)",
        "response": "Una cola es una estructura FIFO (First In, First Out): el primer elemento agregado es el primero en salir. Se implementa con collections.deque o queue.Queue. Operaciones principales: enqueue (agregar al final) y dequeue (remover del inicio). Útil para procesamiento de tareas, BFS, etc."
    },
    {
        "instruction": "¿Qué es una pila (stack)?",
        "response": "Una pila es una estructura LIFO (Last In, First Out): el último elemento agregado es el primero en salir. En Python se implementa con una lista usando append() para push y pop() para pop. Útil para evaluar expresiones, DFS, historial de navegación, y backtracking."
    },
    {
        "instruction": "¿Qué es una lista enlazada?",
        "response": "Una lista enlazada es una estructura donde cada nodo contiene un valor y una referencia al siguiente nodo. A diferencia de arrays, los elementos no están en memoria contigua. Tipos: simple, doble, circular. Ventaja: inserción/eliminación O(1) si tienes la referencia. Desventaja: acceso O(n)."
    },
    {
        "instruction": "Explica qué es un árbol binario",
        "response": "Un árbol binario es una estructura jerárquica donde cada nodo tiene máximo dos hijos: izquierdo y derecho. El nodo superior es la raíz. Si está ordenado (BST), el hijo izquierdo es menor y el derecho mayor. Útil para búsquedas eficientes, expresiones, y jerarquías."
    },
    {
        "instruction": "¿Qué es un heap?",
        "response": "Un heap es un árbol binario completo que satisface la propiedad de heap: en un max-heap, cada padre es mayor o igual a sus hijos; en un min-heap, menor o igual. Se implementa con arrays. Operaciones principales: insert O(log n), extract-max/min O(log n). Python tiene heapq para min-heaps."
    },
    {
        "instruction": "¿Qué es un grafo?",
        "response": "Un grafo es una estructura con nodos (vértices) conectados por aristas (edges). Puede ser dirigido o no dirigido, ponderado o no. Se representa con listas de adyacencia o matriz de adyacencia. Usado en redes sociales, mapas, optimización de rutas, y más."
    },
    {
        "instruction": "Explica qué es una tabla hash",
        "response": "Una tabla hash (hash table) mapea claves a valores usando una función hash. Ofrece acceso O(1) promedio. Los diccionarios de Python son tablas hash. Las colisiones se manejan con chaining (listas) o open addressing. Ideal para búsquedas rápidas y conteo de frecuencias."
    },
    {
        "instruction": "¿Qué es un trie?",
        "response": "Un trie (árbol de prefijos) es una estructura donde cada nodo representa un carácter. Se usa para almacenar strings eficientemente. Ideal para autocompletado, diccionarios, y búsqueda de prefijos. Comparte prefijos comunes, ahorrando espacio. Búsqueda de palabra es O(m) donde m es la longitud de la palabra."
    }
]

dataset.extend(data_structures)

# ========== ALGORITMOS ==========

algorithms = [
    {
        "instruction": "Explica qué es DFS",
        "response": "DFS (Depth-First Search) es un algoritmo de recorrido de grafos que explora lo más profundo posible antes de retroceder. Se implementa con recursión o una pila. Complejidad: O(V+E) donde V son vértices y E aristas. Útil para detectar ciclos, ordenamiento topológico, y resolver laberintos."
    },
    {
        "instruction": "¿Qué es BFS?",
        "response": "BFS (Breadth-First Search) recorre un grafo por niveles, visitando todos los vecinos de un nodo antes de pasar al siguiente nivel. Se implementa con una cola. Complejidad: O(V+E). Útil para encontrar el camino más corto en grafos no ponderados y niveles de un árbol."
    },
    {
        "instruction": "Explica el algoritmo de búsqueda binaria",
        "response": "La búsqueda binaria encuentra un elemento en un array ordenado dividiendo repetidamente el espacio de búsqueda a la mitad. Complejidad: O(log n). Compara el elemento medio: si es igual lo encontró, si es menor busca en la mitad izquierda, si es mayor en la derecha. Requiere que el array esté ordenado."
    },
    {
        "instruction": "¿Qué es el algoritmo de ordenamiento burbuja?",
        "response": "Bubble sort compara pares adyacentes de elementos y los intercambia si están en orden incorrecto. Repite este proceso hasta que no haya más intercambios. Complejidad: O(n²) en peor y promedio caso, O(n) en mejor caso. Es simple pero ineficiente para listas grandes."
    },
    {
        "instruction": "Explica qué es quicksort",
        "response": "Quicksort es un algoritmo de ordenamiento divide y vencerás. Elige un pivote, particiona el array en elementos menores y mayores que el pivote, y ordena recursivamente cada parte. Complejidad promedio: O(n log n), peor caso: O(n²). Es uno de los algoritmos más rápidos en la práctica."
    },
    {
        "instruction": "¿Qué es mergesort?",
        "response": "Mergesort divide el array en mitades, las ordena recursivamente, y luego mezcla las mitades ordenadas. Complejidad: siempre O(n log n). Es estable (mantiene el orden relativo de elementos iguales) y predecible. Desventaja: usa O(n) espacio adicional para el merging."
    },
    {
        "instruction": "Explica el algoritmo de Dijkstra",
        "response": "Dijkstra encuentra el camino más corto desde un nodo fuente a todos los demás en un grafo con pesos no negativos. Usa un min-heap para procesar nodos por distancia. Complejidad: O((V+E) log V) con heap. Mantiene distancias tentativas y las actualiza cuando encuentra rutas mejores."
    },
    {
        "instruction": "¿Qué es programación dinámica?",
        "response": "La programación dinámica resuelve problemas dividiéndolos en subproblemas superpuestos y almacenando resultados (memoización). Evita recalcular los mismos subproblemas. Ejemplos: Fibonacci, problema de la mochila, subsecuencia común más larga. Dos enfoques: top-down (recursión + memo) y bottom-up (iterativo)."
    },
    {
        "instruction": "Explica qué es el algoritmo greedy",
        "response": "Un algoritmo greedy toma la decisión localmente óptima en cada paso esperando llegar a un óptimo global. No siempre garantiza la solución óptima. Ejemplos: algoritmo de Huffman, selección de actividades, Prim y Kruskal. Es simple y eficiente pero no funciona para todos los problemas."
    },
    {
        "instruction": "¿Qué es backtracking?",
        "response": "Backtracking es una técnica que prueba soluciones incrementalmente y retrocede cuando una solución parcial no puede completarse. Es como DFS con restricciones. Ejemplos: N-Queens, Sudoku, generación de permutaciones. Poda ramas que no llevan a soluciones válidas, mejorando eficiencia sobre fuerza bruta."
    },
    {
        "instruction": "Explica el algoritmo de Kruskal",
        "response": "Kruskal encuentra el árbol de expansión mínima (MST) de un grafo. Ordena las aristas por peso y las agrega una por una si no forman ciclos (usando Union-Find). Complejidad: O(E log E) dominada por el ordenamiento. Greedy: siempre elige la arista más barata válida."
    },
    {
        "instruction": "¿Qué es el algoritmo A*?",
        "response": "A* es un algoritmo de búsqueda de caminos que usa una heurística para encontrar el camino más corto eficientemente. Combina el costo real g(n) con una estimación heurística h(n) al objetivo. f(n) = g(n) + h(n). Con una heurística admisible, garantiza encontrar el camino óptimo. Usado en videojuegos y navegación."
    }
]

dataset.extend(algorithms)

# ========== CONCEPTOS AVANZADOS ==========

advanced = [
    {
        "instruction": "¿Qué es la programación asíncrona?",
        "response": "La programación asíncrona permite ejecutar operaciones sin bloquear el programa. En Python se usa async/await. Una función async puede 'await' otras operaciones asíncronas. Útil para I/O: peticiones HTTP, lectura de archivos, consultas DB. Event loops gestionan la ejecución. Mejora el rendimiento en operaciones que esperan."
    },
    {
        "instruction": "Explica qué es un closure",
        "response": "Un closure es una función que recuerda el entorno donde fue creada, incluso después de que ese entorno haya terminado. En Python, una función interna puede acceder a variables de la función externa. Ejemplo: def exterior(): x = 10; def interior(): return x; return interior. La función interior 'cierra' sobre x."
    },
    {
        "instruction": "¿Qué es la metaprogramación?",
        "response": "La metaprogramación es código que manipula código. En Python incluye decoradores, metaclases, y funciones que crean funciones. Permite crear abstracciones poderosas y reducir boilerplate. Ejemplo: un decorador que automáticamente registra funciones. Las metaclases controlan la creación de clases. Es poderoso pero puede hacer el código difícil de entender."
    },
    {
        "instruction": "¿Qué son las metaclases?",
        "response": "Las metaclases son 'clases de clases', controlan cómo se crean las clases. type es la metaclase por defecto. Se heredan con class MiClase(metaclass=MiMeta). Pueden modificar atributos de clase, validar definiciones, o agregar métodos automáticamente. Uso avanzado: ORMs, frameworks. Generalmente hay alternativas más simples."
    },
    {
        "instruction": "Explica qué es el Context Manager",
        "response": "Un context manager gestiona recursos asegurando su correcta adquisición y liberación. Se usa con 'with': with open('file.txt') as f: data = f.read(). Implementa __enter__ y __exit__. También se crea con contextlib.contextmanager. Garantiza limpieza incluso si hay excepciones. Útil para archivos, conexiones DB, locks."
    },
    {
        "instruction": "¿Qué es el GIL en Python?",
        "response": "El GIL (Global Interpreter Lock) es un mutex que protege el acceso a objetos Python, permitiendo que solo un thread ejecute bytecode a la vez. Esto significa que threads en CPython no logran verdadero paralelismo en código CPU-bound. Para paralelismo real se usan procesos (multiprocessing). El GIL no afecta programas I/O-bound."
    },
    {
        "instruction": "Explica qué es la serialización",
        "response": "La serialización convierte objetos en un formato que puede almacenarse o transmitirse, y luego reconstruirse. En Python: pickle para objetos Python, json para datos compatibles con JSON. Útil para guardar estado, comunicación entre procesos, APIs. Ejemplo: json.dumps(obj) convierte a JSON, json.loads(str) reconstruye."
    },
    {
        "instruction": "¿Qué es un descriptor en Python?",
        "response": "Un descriptor es un objeto que define cómo se accede a un atributo. Implementa __get__, __set__, o __delete__. Son la base de properties, métodos, staticmethod, classmethod. Permite controlar acceso a atributos: validación, lazy loading, etc. Ejemplo: property es un descriptor que convierte métodos en atributos."
    },
    {
        "instruction": "¿Qué es el monkey patching?",
        "response": "Monkey patching es modificar o extender código en runtime, típicamente cambiando atributos de clases/módulos. Ejemplo: MyClass.method = new_method. Útil para testing (mocking) o arreglar bugs de terceros. Sin embargo, puede causar comportamiento inesperado y dificultar el debugging. Usar con precaución."
    },
    {
        "instruction": "Explica qué es la inyección de dependencias",
        "response": "La inyección de dependencias es un patrón donde un objeto recibe sus dependencias desde afuera en lugar de crearlas internamente. Mejora testabilidad y desacoplamiento. Ejemplo: en lugar de db = Database() dentro de una clase, se pasa como parámetro: __init__(self, db). Facilita usar mocks en tests."
    },
    {
        "instruction": "¿Qué es un algoritmo de hash criptográfico?",
        "response": "Un hash criptográfico convierte datos de cualquier tamaño en una cadena de tamaño fijo de forma irreversible. Propiedades: mismo input siempre produce mismo output, mínimo cambio en input cambia radicalmente el hash, computacionalmente imposible obtener el input desde el hash. Ejemplos: SHA-256, SHA-3. Usado para passwords, integridad de datos, blockchain."
    },
    {
        "instruction": "Explica el patrón Singleton",
        "response": "Singleton es un patrón de diseño que garantiza que una clase tenga solo una instancia y proporciona un punto de acceso global a ella. En Python se implementa con __new__ o usando un decorador. Útil para gestores de configuración, loggers, pools de conexiones. Crítica: puede ser un anti-patrón si se abusa, complica testing."
    },
    {
        "instruction": "¿Qué es el patrón Factory?",
        "response": "El patrón Factory es un método que crea objetos sin especificar la clase exacta. Delega la instanciación a subclases o métodos. Ejemplo: una función create_shape(type) que retorna Circle(), Square(), etc. según el parámetro. Promueve el código flexible y desacoplado. Útil cuando la creación es compleja."
    },
    {
        "instruction": "Explica qué es el patrón Observer",
        "response": "El patrón Observer define una dependencia uno-a-muchos donde múltiples objetos (observers) se suscriben a cambios de un objeto (subject). Cuando el subject cambia, notifica a todos los observers. Usado en sistemas de eventos, MVC, pub/sub. Ejemplo: un botón notifica a múltiples handlers cuando es clicado."
    },
    {
        "instruction": "¿Qué es un sistema de tipos estático vs dinámico?",
        "response": "En tipado estático (C++, Java) los tipos se verifican en compilación, detectando errores temprano. En tipado dinámico (Python, JavaScript) los tipos se verifican en runtime, ofreciendo más flexibilidad pero detectando errores más tarde. Python tiene type hints opcionales para documentación y análisis estático con mypy sin enforcement en runtime."
    }
]

dataset.extend(advanced)

# ========== BASES DE DATOS Y APIs ==========

databases_apis = [
    {
        "instruction": "¿Qué es SQL?",
        "response": "SQL (Structured Query Language) es un lenguaje para gestionar bases de datos relacionales. Permite crear tablas, insertar datos (INSERT), consultar (SELECT), actualizar (UPDATE), y eliminar (DELETE). Ejemplo: SELECT * FROM usuarios WHERE edad > 18. Soporta joins, agregaciones, subconsultas. Es declarativo: describes qué quieres, no cómo obtenerlo."
    },
    {
        "instruction": "Explica qué es una base de datos NoSQL",
        "response": "Las bases NoSQL no usan el modelo relacional tradicional. Tipos: documentos (MongoDB), clave-valor (Redis), columnas (Cassandra), grafos (Neo4j). Ventajas: escalabilidad horizontal, esquema flexible, mejor rendimiento para ciertos casos. Se usan cuando hay datos no estructurados, alta disponibilidad."
    },
    {
        "instruction": "¿Qué es una API REST?",
        "response": "Una API REST usa HTTP para comunicación entre sistemas. Principios: recursos identificados por URLs, uso de métodos HTTP (GET, POST, PUT, DELETE), sin estado (stateless), respuestas típicamente en JSON. Ejemplo: GET /api/usuarios/123 obtiene un usuario. Es el estándar más común para APIs web."
    },
    {
        "instruction": "¿Qué es un ORM?",
        "response": "Un ORM (Object-Relational Mapping) mapea objetos de programación a tablas de base de datos. En Python: SQLAlchemy, Django ORM. Permite trabajar con BD usando objetos en lugar de SQL directo. Ejemplo: user = User.query.filter_by(name='Juan').first(). Ventaja: abstracción y portabilidad. Desventaja: puede ser menos eficiente que SQL optimizado."
    },
    {
        "instruction": "Explica qué es una transacción en BD",
        "response": "Una transacción es un conjunto de operaciones que se ejecutan como una unidad atómica: todas se completan o ninguna. Propiedades ACID: Atomicidad, Consistencia, Aislamiento, Durabilidad. Se inicia con BEGIN, se confirma con COMMIT o se revierte con ROLLBACK. Garantiza integridad de datos en operaciones críticas como transferencias bancarias."
    },
    {
        "instruction": "¿Qué es un índice en bases de datos?",
        "response": "Un índice es una estructura de datos que mejora la velocidad de búsqueda en una tabla. Similar a un índice de libro. Acelera consultas SELECT pero ralentiza INSERT/UPDATE/DELETE. Se crea con CREATE INDEX. Tipos: B-tree (por defecto), hash, full-text. Usar en columnas frecuentemente buscadas o usadas en WHERE/JOIN."
    },
    {
        "instruction": "¿Qué es normalización de bases de datos?",
        "response": "Normalización organiza datos para reducir redundancia y dependencias. Formas normales: 1NF (valores atómicos), 2NF (sin dependencias parciales), 3NF (sin dependencias transitivas). Ejemplo: separar tabla de pedidos y productos en lugar de repetir datos de producto en cada pedido. Mejora integridad pero puede requerir más joins."
    },
    {
        "instruction": "Explica qué es un JOIN en SQL",
        "response": "JOIN combina filas de dos o más tablas basándose en una columna relacionada. INNER JOIN: solo coincidencias. LEFT JOIN: todas de izquierda + coincidencias de derecha. RIGHT JOIN: opuesto a LEFT. FULL JOIN: todas las filas. Ejemplo: SELECT * FROM usuarios INNER JOIN pedidos ON usuarios.id = pedidos.user_id."
    },
    {
        "instruction": "¿Qué es GraphQL?",
        "response": "GraphQL es un lenguaje de consulta para APIs creado por Facebook. El cliente especifica exactamente qué datos necesita, evitando over-fetching o under-fetching. Una sola petición puede obtener datos relacionados. Usa un schema tipado. Ejemplo: query { user(id: 123) { name, posts { title } } }. Alternativa más flexible a REST."
    },
    {
        "instruction": "¿Qué es Redis?",
        "response": "Redis es una base de datos NoSQL en memoria de tipo clave-valor. Extremadamente rápida (microsegundos). Soporta estructuras: strings, hashes, listas, sets, sorted sets. Usado para caché, sesiones, colas, real-time. Puede persistir datos en disco. Operaciones atómicas. Ejemplo: SET usuario:123 'Juan', GET usuario:123."
    },
    {
        "instruction": "Explica qué es MongoDB",
        "response": "MongoDB es una base de datos NoSQL orientada a documentos. Almacena datos en formato BSON (similar a JSON). Esquema flexible: documentos en la misma colección pueden tener diferentes campos. Escalable horizontalmente con sharding. Consultas ricas y agregaciones. Ejemplo: db.usuarios.find({edad: {$gt: 18}}). Ideal para datos semi-estructurados."
    }
]

dataset.extend(databases_apis)

# ========== WEB DEVELOPMENT ==========

web_dev = [
    {
        "instruction": "¿Qué es HTTP?",
        "response": "HTTP (HyperText Transfer Protocol) es el protocolo de comunicación web. Cliente (navegador) hace peticiones, servidor responde. Métodos: GET (obtener), POST (crear), PUT (actualizar), DELETE (eliminar). Status codes: 200 OK, 404 Not Found, 500 Server Error. HTTPS es la versión segura con encriptación SSL/TLS."
    },
    {
        "instruction": "Explica qué es una cookie",
        "response": "Una cookie es un pequeño archivo que un servidor envía al navegador para almacenar información. El navegador la envía de vuelta en peticiones futuras. Usos: sesiones, preferencias, tracking. Tipos: session cookies (temporales) y persistent cookies (con expiración). Se gestionan con encabezados Set-Cookie y Cookie."
    },
    {
        "instruction": "¿Qué es CORS?",
        "response": "CORS (Cross-Origin Resource Sharing) es un mecanismo que permite que recursos de un dominio sean solicitados desde otro dominio. Por seguridad, los navegadores bloquean peticiones cross-origin por defecto. El servidor debe incluir encabezados Access-Control-Allow-Origin para permitirlas. Previene ataques pero puede causar problemas en desarrollo."
    },
    {
        "instruction": "¿Qué es JWT?",
        "response": "JWT (JSON Web Token) es un estándar para crear tokens de acceso. Consta de: header (algoritmo), payload (datos), signature (verificación). Se codifica en Base64. Usado para autenticación stateless: el servidor no guarda sesión, solo verifica la firma. Ejemplo: Authorization: Bearer <token>. Expira después de cierto tiempo."
    },
    {
        "instruction": "Explica qué es un WebSocket",
        "response": "WebSocket es un protocolo que permite comunicación bidireccional full-duplex entre cliente y servidor sobre una única conexión persistente. A diferencia de HTTP (request-response), permite que el servidor envíe datos sin que el cliente lo solicite. Ideal para chat, notificaciones en tiempo real, juegos online. Inicia con handshake HTTP, luego actualiza a WebSocket."
    },
    {
        "instruction": "¿Qué es el patrón MVC?",
        "response": "MVC (Model-View-Controller) separa una aplicación en tres componentes: Model (datos y lógica de negocio), View (presentación/UI), Controller (maneja input y coordina). Promueve separación de responsabilidades. Ejemplo: usuario hace clic -> Controller procesa -> actualiza Model -> View se renderiza. Usado en frameworks como Django, Rails."
    },
    {
        "instruction": "¿Qué es un middleware?",
        "response": "Un middleware es código que se ejecuta entre la petición y la respuesta. Puede modificar request/response, ejecutar código, terminar el request-response cycle. Ejemplos: autenticación, logging, CORS, compresión. En Flask/Django se apilan secuencialmente. Permite agregar funcionalidad de manera modular y reutilizable."
    },
    {
        "instruction": "Explica qué es Server-Side Rendering",
        "response": "SSR (Server-Side Rendering) genera HTML en el servidor y lo envía completo al cliente. Ventajas: mejor SEO, tiempo a primer render más rápido, funciona sin JS. Desventaja: carga en servidor, navegación puede ser más lenta. Opuesto a CSR (Client-Side Rendering) donde JS genera el HTML. Frameworks: Next.js, Nuxt.js."
    },
    {
        "instruction": "¿Qué es un CDN?",
        "response": "Un CDN (Content Delivery Network) es una red de servidores distribuidos geográficamente que entrega contenido estático (imágenes, CSS, JS) desde el servidor más cercano al usuario. Reduce latencia, mejora velocidad de carga, reduce carga en servidor origen. Ejemplos: Cloudflare, Akamai. Común para librerías populares y assets de sitios grandes."
    },
    {
        "instruction": "¿Qué es el lazy loading?",
        "response": "Lazy loading es cargar recursos solo cuando son necesarios, en lugar de todos al inicio. Común para imágenes: se cargan cuando están por aparecer en viewport. Mejora tiempo de carga inicial y ahorra ancho de banda. En código: import dinámico de módulos. HTML: <img loading='lazy'>. Frameworks SPA lo usan para code splitting de rutas."
    }
]

dataset.extend(web_dev)

# ========== TESTING Y DEBUGGING ==========

testing = [
    {
        "instruction": "¿Qué es unit testing?",
        "response": "Unit testing prueba unidades individuales de código (funciones, métodos) de forma aislada. Se usan frameworks como pytest o unittest. Cada test verifica un comportamiento específico. Ejemplo: test_suma() verifica que suma(2,3) == 5. Buenos tests son rápidos, independientes, y repetibles. Mejoran confianza al refactorizar."
    },
    {
        "instruction": "¿Qué es TDD?",
        "response": "TDD (Test-Driven Development) es escribir tests antes del código. Ciclo: Red (escribir test que falla), Green (escribir código mínimo para pasar), Refactor (mejorar código). Beneficios: mejor diseño, cobertura completa, documentación viva. Requiere disciplina pero resulta en código más limpio y menos bugs."
    },
    {
        "instruction": "Explica qué es integration testing",
        "response": "Integration testing verifica que múltiples componentes trabajen juntos correctamente. Prueba interfaces entre módulos, llamadas a BD, APIs externas. Más lentos que unit tests pero detectan problemas de integración. Ejemplo: probar que un endpoint guarde correctamente en BD. Se complementan con unit tests y end-to-end tests."
    },
    {
        "instruction": "¿Qué es mocking?",
        "response": "Mocking es crear objetos falsos que simulan el comportamiento de objetos reales. Usado en tests para aislar código y evitar dependencias externas (BD, APIs). Ejemplo: mock de una API que siempre retorna datos predefinidos. Python: unittest.mock. Permite probar casos difíciles como errores de red sin depender de servicios externos."
    },
    {
        "instruction": "¿Qué es code coverage?",
        "response": "Code coverage mide qué porcentaje del código es ejecutado por los tests. Herramientas: coverage.py, pytest-cov. Tipos: line coverage (líneas ejecutadas), branch coverage (ramas condicionales). 100% coverage no garantiza ausencia de bugs, pero low coverage indica áreas no probadas. Meta típica: 80-90%."
    },
    {
        "instruction": "Explica qué es el debugging",
        "response": "Debugging es encontrar y corregir errores en el código. Técnicas: usar debugger (pdb en Python, permite ejecutar paso a paso), print statements, logging, rubber duck debugging. Debugger permite inspeccionar variables, establecer breakpoints. Importante: reproducir el bug, aislar la causa, verificar la solución, agregar test para prevenir regresión."
    },
    {
        "instruction": "¿Qué es un breakpoint?",
        "response": "Un breakpoint es un punto en el código donde el debugger pausa la ejecución. Permite inspeccionar el estado del programa en ese momento: valores de variables, call stack. Se establecen en IDEs o con pdb.set_trace() en Python. Útil para entender flujo de ejecución y encontrar dónde ocurre un bug."
    },
    {
        "instruction": "¿Qué es el logging?",
        "response": "Logging es registrar eventos durante la ejecución del programa. A diferencia de print, tiene niveles (DEBUG, INFO, WARNING, ERROR, CRITICAL), puede dirigirse a archivos, incluye timestamps. Python: import logging. Ejemplo: logging.error('Falló conexión'). Esencial para debugging en producción y auditoría. Configurable para diferentes ambientes."
    },
    {
        "instruction": "Explica qué es CI/CD",
        "response": "CI (Continuous Integration) integra cambios frecuentemente, ejecutando tests automáticamente. CD (Continuous Deployment/Delivery) automatiza el despliegue a producción. Pipeline típico: commit -> build -> test -> deploy. Herramientas: Jenkins, GitHub Actions, GitLab CI. Detecta bugs temprano, reduce riesgo de despliegue, acelera entrega de features."
    },
    {
        "instruction": "¿Qué es un linter?",
        "response": "Un linter analiza código estáticamente para encontrar errores de estilo, bugs potenciales, y code smells sin ejecutarlo. Python: pylint, flake8, ruff. Verifica: convenciones de nombrado, imports no usados, complejidad. Configurable con archivos .pylintrc. Se integra en IDE y CI/CD. Mantiene código consistente y de calidad en equipos."
    }
]

dataset.extend(testing)

# ========== SEGURIDAD ==========

security = [
    {
        "instruction": "¿Qué es SQL Injection?",
        "response": "SQL Injection es inyectar código SQL malicioso en inputs para manipular BD. Ejemplo: si username viene de usuario sin validar: SELECT * FROM users WHERE name = '{username}' -> si username es ' OR '1'='1 retorna todos los usuarios. Prevención: usar prepared statements/parametrizadas, ORM, validar inputs, principio de menor privilegio en BD."
    },
    {
        "instruction": "¿Qué es XSS?",
        "response": "XSS (Cross-Site Scripting) inyecta scripts maliciosos en páginas web vistas por otros usuarios. Tipos: Stored (guardado en BD), Reflected (en URL), DOM-based. Ejemplo: comentario con <script>alert('XSS')</script>. Prevención: escapar HTML output, Content Security Policy, validar inputs, usar frameworks que escapan automáticamente."
    },
    {
        "instruction": "Explica qué es CSRF",
        "response": "CSRF (Cross-Site Request Forgery) engaña al usuario autenticado para ejecutar acciones no deseadas. Ejemplo: email malicioso con <img src='bank.com/transfer?to=atacante&amount=1000'>. El navegador envía cookies automáticamente. Prevención: CSRF tokens (valores aleatorios que el servidor verifica), SameSite cookies, verificar Origin/Referer headers."
    },
    {
        "instruction": "¿Qué es hashing de passwords?",
        "response": "Hashing convierte passwords en strings irreversibles antes de guardarlos. Si la BD es comprometida, los passwords originales permanecen secretos. Usar algoritmos diseñados para passwords: bcrypt, Argon2, scrypt (no SHA-256/MD5). Incluyen salt (valor aleatorio) automáticamente. Ejemplo Python: bcrypt.hashpw(password, bcrypt.gensalt())."
    },
    {
        "instruction": "¿Qué es HTTPS?",
        "response": "HTTPS es HTTP sobre SSL/TLS, encriptando la comunicación entre cliente y servidor. Previene: eavesdropping (escucha), tampering (modificación), impersonation (suplantación). Requiere certificado SSL del servidor. El navegador muestra candado. Esencial para cualquier sitio que maneje datos sensibles. Let's Encrypt ofrece certificados gratuitos."
    },
    {
        "instruction": "Explica qué es OAuth",
        "response": "OAuth es un protocolo de autorización que permite que apps accedan a recursos de usuarios sin compartir contraseñas. Flujo: usuario autoriza en proveedor (Google, GitHub) -> app recibe access token -> usa token para acceder a API. No es autenticación (eso es OpenID Connect). Usado en 'Iniciar sesión con Google'."
    },
    {
        "instruction": "¿Qué es un ataque de fuerza bruta?",
        "response": "Un ataque de fuerza bruta prueba sistemáticamente todas las combinaciones posibles de contraseñas hasta encontrar la correcta. Contramedidas: rate limiting (limitar intentos), account lockout (bloquear tras X intentos fallidos), CAPTCHA, contraseñas fuertes, 2FA. Diccionario attack usa palabras comunes en lugar de todas las combinaciones."
    },
    {
        "instruction": "¿Qué es 2FA?",
        "response": "2FA (Two-Factor Authentication) requiere dos formas de verificación: algo que sabes (password) y algo que tienes (código de app, SMS, hardware key). Incluso si la contraseña es comprometida, el atacante no puede acceder sin el segundo factor. TOTP (Time-based One-Time Password) es común. Aumenta significativamente la seguridad."
    },
    {
        "instruction": "Explica qué es el principio de menor privilegio",
        "response": "El principio de menor privilegio otorga solo los permisos mínimos necesarios para realizar una tarea. Ejemplos: usuario de BD con solo permisos SELECT, no admin root para aplicaciones. Limita el daño potencial de brechas de seguridad. Aplicar en: permisos de archivos, roles de BD, APIs, acceso a servidores."
    },
    {
        "instruction": "¿Qué es sanitización de inputs?",
        "response": "Sanitización es limpiar/validar inputs de usuario para prevenir ataques. Incluye: validar tipo y formato, escapar caracteres especiales, limitar longitud, whitelist de valores permitidos. Nunca confiar en inputs del cliente. Validar tanto en frontend (UX) como backend (seguridad). Usar librerías establecidas en lugar de implementaciones caseras."
    }
]

dataset.extend(security)

# ========== GIT Y CONTROL DE VERSIONES ==========

git = [
    {
        "instruction": "¿Qué es Git?",
        "response": "Git es un sistema de control de versiones distribuido. Rastrea cambios en código, permite colaboración, y mantiene historial completo. Cada desarrollador tiene una copia completa del repositorio. Comandos básicos: git init, git add, git commit, git push, git pull. Creado por Linus Torvalds. Es el estándar de la industria."
    },
    {
        "instruction": "¿Qué es un commit?",
        "response": "Un commit es un snapshot del estado del proyecto en un momento. Incluye: cambios, autor, timestamp, mensaje descriptivo. Se crea con git commit -m 'mensaje'. Cada commit tiene un hash único (SHA). Forman una cadena que representa el historial. Buenos mensajes son concisos pero descriptivos: 'Fix login bug' no 'changes'."
    },
    {
        "instruction": "Explica qué es una rama (branch)",
        "response": "Una rama es una línea independiente de desarrollo. Permite trabajar en features sin afectar main/master. Se crea con git branch nombre, cambias con git checkout nombre. Workflow común: crear rama para feature, hacer commits, merge a main cuando está listo. Ramas permiten trabajo paralelo y experimentación segura."
    },
    {
        "instruction": "¿Qué es un merge?",
        "response": "Merge combina cambios de dos ramas. git merge feature-branch integra cambios de feature-branch a la rama actual. Si no hay conflictos, crea un merge commit automáticamente. Si hay conflictos (cambios en las mismas líneas), Git marca los conflictos y requiere resolución manual. Fast-forward merge mueve el puntero sin crear commit de merge."
    },
    {
        "instruction": "¿Qué es un conflicto de merge?",
        "response": "Un conflicto ocurre cuando dos ramas modifican las mismas líneas de un archivo. Git no puede decidir automáticamente qué cambio mantener. Marca el conflicto con <<<<<<, ======, >>>>>>. Debes editar manualmente para resolver, luego git add y git commit. Prevención: commits frecuentes, comunicación en equipo, pull antes de push."
    },
    {
        "instruction": "Explica qué es git rebase",
        "response": "Rebase reaplica commits de una rama encima de otra, creando un historial lineal. git rebase main mueve tu rama a la punta de main. Diferencia con merge: rebase reescribe historial (cambia SHAs), merge lo preserva. Regla: nunca rebase ramas públicas. Útil para limpiar historial antes de merge. Interactive rebase permite editar commits."
    },
    {
        "instruction": "¿Qué es un pull request?",
        "response": "Un pull request (PR) es una petición para integrar cambios de una rama a otra en plataformas como GitHub. Permite: revisar código, discutir cambios, ejecutar CI, aprobar antes de merge. Workflow: crear rama, hacer cambios, push, abrir PR, revisar, merge. Es fundamental para colaboración y control de calidad en equipos."
    },
    {
        "instruction": "¿Qué es .gitignore?",
        "response": "El archivo .gitignore especifica archivos/directorios que Git debe ignorar. Ejemplo: __pycache__/, .env, *.log. Evita trackear archivos generados, configuraciones locales, credenciales. Se crean patrones por línea. GitHub tiene plantillas para diferentes lenguajes. Cambios en .gitignore no afectan archivos ya trackeados (usar git rm --cached)."
    },
    {
        "instruction": "Explica qué es git stash",
        "response": "git stash guarda temporalmente cambios no commiteados para limpiar el working directory. Útil cuando necesitas cambiar de rama pero no quieres commitear trabajo incompleto. git stash aplica los cambios guardados. git stash list muestra stashes. git stash pop aplica y elimina el stash. Puede guardar múltiples stashes."
    },
    {
        "instruction": "¿Qué es git cherry-pick?",
        "response": "cherry-pick aplica un commit específico de una rama a otra. git cherry-pick <commit-hash> copia ese commit a la rama actual. Útil para aplicar bugfixes selectivamente sin merge completo. Crea un nuevo commit con el mismo cambio pero diferente hash. Usar con moderación para no duplicar historial innecesariamente."
    }
]

dataset.extend(git)

# ========== PERFORMANCE Y OPTIMIZACIÓN ==========

performance = [
    {
        "instruction": "¿Qué es la complejidad espacial?",
        "response": "La complejidad espacial mide cuánta memoria usa un algoritmo respecto al tamaño de entrada. Notación Big O igual que temporal. O(1) usa espacio constante, O(n) espacio proporcional a entrada. Ejemplo: algoritmo recursivo usa O(n) espacio en call stack. Importante balancear tiempo vs espacio: a veces usar más memoria reduce tiempo de ejecución."
    },
    {
        "instruction": "¿Qué es caching?",
        "response": "Caching almacena resultados de operaciones costosas para reutilizarlos. Tipos: in-memory (Redis, Memcached), HTTP cache, aplicación (memoization). Invalida cuando datos cambian. Políticas de eviction: LRU (Least Recently Used), LFU, FIFO. Ejemplo: cachear query de BD, resultados de API. Mejora dramáticamente rendimiento para datos consultados frecuentemente."
    },
    {
        "instruction": "Explica qué es el profiling",
        "response": "Profiling mide dónde gasta tiempo/memoria un programa. Identifica cuellos de botella. Python: cProfile, line_profiler, memory_profiler. Genera reportes con funciones más costosas. Optimización prematura es mala; primero profile, luego optimiza. Enfocarse en el código que realmente impacta (regla 80/20). Herramientas visuales: snakeviz."
    },
    {
        "instruction": "¿Qué es lazy evaluation?",
        "response": "Lazy evaluation pospone cálculos hasta que son necesarios. Contrario a eager evaluation que calcula inmediatamente. Ventajas: evita trabajo innecesario, permite estructuras infinitas. Python: generadores son lazy. Ejemplo: range() en Python 3 es lazy, no crea lista completa. Funcional programming usa lazy evaluation extensivamente."
    },
    {
        "instruction": "¿Qué es memoización?",
        "response": "Memoización cachea resultados de función basándose en sus argumentos. Si la función se llama con mismos args, retorna resultado cacheado. Python: @lru_cache. Ejemplo: Fibonacci recursivo sin memo es O(2^n), con memo es O(n). Trade-off: usa más memoria. Solo para funciones puras (sin side effects, mismo input -> mismo output)."
    },
    {
        "instruction": "Explica qué es database indexing",
        "response": "Indexar es crear estructura de datos (B-tree, hash) para acelerar búsquedas en BD. Similar a índice de libro. Ventaja: queries más rápidos (de O(n) a O(log n)). Desventaja: ralentiza escrituras, usa espacio. Indexar columnas en WHERE, JOIN, ORDER BY. Índices compuestos para queries con múltiples columnas. Analizar query plans para optimizar."
    },
    {
        "instruction": "¿Qué es el query optimization?",
        "response": "Query optimization mejora el rendimiento de consultas SQL. Técnicas: usar índices apropiados, evitar SELECT *, limitar resultados, usar JOINs eficientemente, evitar N+1 queries. Herramientas: EXPLAIN para ver query plan. Considerar denormalización para lecturas frecuentes. A veces crear índices resuelve problemas de rendimiento sin cambiar queries."
    },
    {
        "instruction": "¿Qué es el N+1 query problem?",
        "response": "El problema N+1 ocurre cuando se hace una query para obtener N items y luego 1 query adicional por cada item. Ejemplo: obtener usuarios (1 query), luego para cada usuario obtener sus posts (N queries). Solución: eager loading/JOIN para obtener todo en 1-2 queries. En ORMs: select_related (SQL JOIN) o prefetch_related (consultas separadas optimizadas)."
    },
    {
        "instruction": "Explica qué es load balancing",
        "response": "Load balancing distribuye tráfico entre múltiples servidores para mejorar rendimiento y disponibilidad. Algoritmos: round-robin, least connections, IP hash. Si un servidor falla, el balanceador redirige a otros. Hardware o software (Nginx, HAProxy). Permite escalamiento horizontal. Session stickiness mantiene usuario en mismo servidor para sesiones."
    },
    {
        "instruction": "¿Qué es el horizontal vs vertical scaling?",
        "response": "Scaling vertical agrega más recursos (CPU, RAM) a un servidor existente. Límite: capacidad máxima de hardware. Scaling horizontal agrega más servidores. Ventajas de horizontal: casi ilimitado, mejor disponibilidad (redundancia). Desventajas: más complejo, requiere balanceador, manejo de estado distribuido. Bases NoSQL diseñadas para horizontal scaling."
    }
]

dataset.extend(performance)

# ========== PATRONES DE DISEÑO ADICIONALES ==========

more_patterns = [
    {
        "instruction": "¿Qué es el patrón Strategy?",
        "response": "Strategy define una familia de algoritmos intercambiables. Cada algoritmo está encapsulado y se puede cambiar en runtime. Ejemplo: diferentes estrategias de ordenamiento que se pueden intercambiar. Evita condicionales largos. Componentes: Context (usa estrategia), Strategy interface, ConcreteStrategies. Promueve Open/Closed Principle."
    },
    {
        "instruction": "Explica el patrón Decorator",
        "response": "Decorator agrega funcionalidad a objetos dinámicamente sin modificar su clase. Envuelve el objeto original. Python: decoradores de funciones (@) son un ejemplo. Ejemplo: agregar scroll, border, shadow a una ventana combinando decoradores. Alternativa flexible a herencia. Mantiene Single Responsibility: cada decorador hace una cosa."
    },
    {
        "instruction": "¿Qué es el patrón Adapter?",
        "response": "Adapter convierte la interfaz de una clase en otra que el cliente espera. Permite que clases incompatibles trabajen juntas. Ejemplo: adaptar API de terceros a tu interfaz. Wrapper que traduce llamadas. Útil al integrar código legacy o librerías externas sin modificarlas. También llamado Wrapper."
    },
    {
        "instruction": "¿Qué es el patrón Template Method?",
        "response": "Template Method define el esqueleto de un algoritmo en una clase base, permitiendo que subclases overriden pasos específicos. La estructura general es fija, detalles varían. Ejemplo: algoritmo de preparar bebida (hervir agua es fijo, agregar ingrediente varía). Usa hooks para opcionalidad. Invierte control: framework llama a tu código."
    },
    {
        "instruction": "Explica el patrón Repository",
        "response": "Repository abstrae el acceso a datos, actuando como una colección en memoria. Desacopla lógica de negocio de persistencia. Ejemplo: UserRepository con métodos find_by_id, save, delete. La implementación real puede ser SQL, NoSQL, memoria. Facilita testing con repositorios falsos. Común en arquitecturas limpias y DDD."
    }
]

dataset.extend(more_patterns)

# ========== MACHINE LEARNING BÁSICO ==========

ml_basics = [
    {
        "instruction": "¿Qué es Machine Learning?",
        "response": "Machine Learning es un subcampo de IA donde sistemas aprenden patrones de datos sin ser programados explícitamente. Tipos: supervised (con labels), unsupervised (sin labels), reinforcement (por recompensas). Proceso: recolectar datos, preprocesar, entrenar modelo, evaluar, desplegar. Librerías Python: scikit-learn, TensorFlow, PyTorch."
    },
    {
        "instruction": "¿Qué es entrenamiento y validación?",
        "response": "El dataset se divide en: entrenamiento (60-80%, para aprender), validación (10-20%, para ajustar hiperparámetros), test (10-20%, para evaluación final). Evita overfitting: modelo que memoriza entrenamiento pero falla en datos nuevos. Cross-validation divide datos en k folds, entrena k veces alternando qué fold es validación. Nunca tocar test set hasta el final."
    },
    {
        "instruction": "Explica qué es regresión lineal",
        "response": "Regresión lineal modela relación entre variables con una línea: y = mx + b. Predice valores continuos. Objetivo: minimizar error cuadrático medio (MSE) entre predicciones y valores reales. Usado en predicción de precios, tendencias. Supone relación lineal, homocedasticidad, independencia de errores. Implementación simple en scikit-learn: LinearRegression().fit(X_train, y_train)."
    },
    {
        "instruction": "¿Qué es clasificación?",
        "response": "Clasificación asigna etiquetas discretas a datos. Ejemplos: spam vs no spam, diagnóstico médico. Algoritmos comunes: regresión logística, árboles de decisión, SVM, redes neuronales. Métricas: accuracy, precision, recall, F1-score. Datos etiquetados son necesarios (supervised learning). Problemas multiclase (más de 2 clases) y binarios (2 clases)."
    },
    {
        "instruction": "¿Qué es overfitting?",
        "response": "Overfitting ocurre cuando un modelo aprende demasiado bien los datos de entrenamiento, incluyendo ruido, y falla en generalizar a datos nuevos. Síntomas: alta precisión en entrenamiento pero baja en validación/test. Prevención: más datos, regularización (L1/L2), dropout en redes neuronales, early stopping, cross-validation. Modelo debe capturar patrones generales."
    },
    {
        "instruction": "Explica qué es una red neuronal",
        "response": "Una red neuronal es un modelo inspirado en el cerebro humano, compuesto por capas de nodos (neuronas). Cada neurona recibe entradas, las pondera, aplica una función de activación y pasa la salida a la siguiente capa. Usada para tareas complejas: visión por computadora, NLP. Arquitecturas comunes: feedforward, convolucional (CNN), recurrente (RNN). Entrenadas con backpropagation."
    },
    {
        "instruction": "¿Qué es el aprendizaje supervisado?",
        "response": "Aprendizaje supervisado usa datos etiquetados para entrenar modelos. El modelo aprende a mapear entradas a salidas correctas. Ejemplos: clasificación (etiquetas discretas), regresión (valores continuos). Proceso: alimentar datos de entrenamiento, ajustar parámetros para minimizar error. Requiere gran cantidad de datos etiquetados para buen rendimiento."
    },
    {
        "instruction": "¿Qué es el aprendizaje no supervisado?",
        "response": "Aprendizaje no supervisado trabaja con datos sin etiquetas. El modelo encuentra patrones o estructuras inherentes. Ejemplos: clustering (K-means, DBSCAN), reducción de dimensionalidad (PCA). Usado para segmentación de clientes, detección de anomalías. No hay una 'respuesta correcta', por lo que la evaluación es más subjetiva."
    },
    {
        "instruction": "Explica qué es la validación cruzada",
        "response": "Validación cruzada (cross-validation) evalúa el rendimiento del modelo dividiendo datos en k subsets (folds). El modelo se entrena k veces, cada vez usando un fold diferente como validación y los otros k-1 como entrenamiento. Promedio de resultados da una estimación robusta del rendimiento. Reduce varianza y ayuda a detectar overfitting."
    },
    {
        "instruction": "¿Qué es la regularización?",
        "response": "Regularización agrega una penalización al error del modelo para evitar overfitting. Tipos comunes: L1 (Lasso) que puede eliminar características, L2 (Ridge) que reduce magnitudes de coeficientes. Controla complejidad del modelo. En redes neuronales, dropout desactiva aleatoriamente neuronas durante entrenamiento. Ayuda a generalizar mejor a datos nuevos."
    },
]

dataset.extend(ml_basics)

# Guardar dataset en archivo JSONL
with open("tutor_programacion.jsonl", "w", encoding="utf-8") as f:
    for entry in dataset:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

print(f"Dataset guardado exitosamente con {len(dataset)} ejemplos")
