import csv
import random
from datetime import datetime, timedelta

# Componentes REALISTAS para Generación Z
frases_gen_z = [
    # Sobre productividad y burnout
    ["Llevo 3 días sin dormir bien", "la presión de ser productivo 24/7", "me tiene al borde del colapso", "¿cuándo normalizamos el burnout?"],
    ["Todos mis amigos están", "emprendiendo o haciendo side hustles", "y yo solo quiero", "un trabajo que no me consuma la vida"],
    ["Mi feed está lleno de", "gente de mi edad siendo CEOs", "mientras yo", "apenas puedo pagar la renta"],
    ["¿Por qué tengo que", "monetizar mis hobbies", "para que valgan la pena?", "Solo quiero dibujar sin pensar en vender"],
    
    # Sobre redes sociales e identidad
    ["Pasé 2 horas editando", "una foto para Instagram", "que borré a los 10 minutos", "porque no tuvo suficientes likes"],
    ["TikTok me tiene", "con una capacidad de atención", "de 15 segundos", "ya ni puedo ver películas completas"],
    ["Subo stories feliz pero", "en realidad estoy", "pasándola horrible", "las redes son puro teatro"],
    ["Dejé Instagram por una semana", "y me di cuenta de", "cuánto tiempo perdía", "comparándome con desconocidos"],
    ["El algoritmo decide", "qué veo, qué pienso, qué compro", "y ni siquiera", "me había dado cuenta"],
    
    # Sobre crisis existencial
    ["Tengo 25 años y", "siento que debería tener", "mi vida resuelta", "pero no sé ni qué quiero hacer"],
    ["Estudié 5 años para", "un título que no me sirve", "en un mercado laboral", "que ya no existe"],
    ["Todos hablan del", "cambio climático", "pero nadie hace nada real", "¿para qué planear un futuro?"],
    ["Me siento perdido", "entre lo que esperan de mí", "y lo que realmente quiero", "la presión es insoportable"],
    
    # Sobre trabajo y economía
    ["Tres trabajos y", "aún no me alcanza", "para independizarme", "¿así va a ser toda la vida?"],
    ["Me piden 5 años de experiencia", "para un trabajo entry level", "que paga salario mínimo", "el mercado laboral está roto"],
    ["Mis papás compraron casa", "a mi edad", "yo apenas puedo", "pagar mi comida del mes"],
    ["La cultura del hustle", "nos hizo creer que", "no descansar es éxito", "pero solo estamos agotados"],
    
    # Sobre salud mental
    ["Normalizar la terapia", "fue lo mejor que nos pasó", "pero también expuso", "que todos estamos rotos"],
    ["Mi ansiedad tiene ansiedad", "y encima siento culpa", "por no estar bien", "es un círculo sin fin"],
    ["Todos mis amigos", "están en terapia o medicados", "y aún así", "seguimos funcionando mal"],
    ["La salud mental es importante", "hasta que pides días libres", "por ansiedad", "y te ven como débil"],
    
    # Sobre tecnología y conexión
    ["Tengo 500 contactos", "en WhatsApp", "pero me siento", "más solo que nunca"],
    ["Las videollamadas", "no reemplazan", "el contacto humano real", "extraño abrazos sin pantallas"],
    ["Zoom, Teams, Discord", "mi vida social", "es 100% digital", "¿cuándo fue la última vez que salí?"],
    ["Chateamos todo el día", "pero cuando nos vemos", "no sabemos de qué hablar", "perdimos la conexión real"],
]

# Más variaciones
opiniones_gen_z = [
    ["honestamente", "ya no sé qué es real", "todo es contenido", "todo es performance"],
    ["estoy cansado de", "fingir que todo está bien", "cuando claramente", "nada lo está"],
    ["la cultura de internet", "destruyó nuestra capacidad", "de estar aburridos", "y ahí está la creatividad"],
    ["crecimos con YouTube", "creyendo que todos", "podíamos ser famosos", "qué decepción"],
    ["mi rutina es", "scroll, trabajo, scroll, dormir", "y repetir", "¿esto es vivir?"],
    ["entre el doom scrolling", "y las noticias depresivas", "mi salud mental", "va en picada"],
    ["antes teníamos Messenger", "y éramos felices", "ahora hay 20 apps", "y nadie responde"],
    ["mi cerebro necesita", "estímulos constantes", "gracias TikTok", "por romper mi atención"],
]

# Componentes para IA
frases_ia = [
    # Sobre dependencia
    ["ChatGPT hace mi tarea", "Google Maps decide mi ruta", "Spotify elige mi música", "¿todavía pienso por mí mismo?"],
    ["Le pregunto a la IA", "cosas que antes", "resolvía pensando", "estoy perdiendo mi criterio"],
    ["Los algoritmos saben", "más de mí", "que yo mismo", "es perturbador y conveniente"],
    ["Alexa controla mi casa", "mi celular mi vida", "la IA mi trabajo", "¿dónde estoy yo en todo esto?"],
    
    # Sobre trabajo y creatividad
    ["La IA genera arte", "en 5 segundos", "yo tardo horas", "¿para qué seguir creando?"],
    ["Mi trabajo ahora es", "revisar lo que hace la IA", "básicamente soy", "un supervisor de máquinas"],
    ["La IA escribe mejor que yo", "diseña mejor que yo", "y es más rápida", "¿qué me queda?"],
    ["Empresas reemplazando humanos", "con IA y bots", "llamándolo", "optimización de recursos"],
    
    # Sobre decisiones y autonomía
    ["Netflix decide qué veo", "Amazon qué compro", "Instagram qué me gusta", "perdí mi libre albedrío"],
    ["El algoritmo me muestra", "solo lo que confirma", "mis creencias", "vivo en una burbuja digital"],
    ["Tinder decide mis matches", "LinkedIn mis trabajos", "las apps", "controlan mi vida social y laboral"],
    ["La IA predice", "mis deseos antes", "de que los tenga", "¿soy tan predecible?"],
    
    # Sobre privacidad y control
    ["Cada clic, cada búsqueda", "alimenta al algoritmo", "que me conoce mejor", "que mi familia"],
    ["Renuncié a mi privacidad", "por comodidad", "y ahora", "no sé cómo recuperarla"],
    ["Las apps escuchan", "mis conversaciones", "porque me aparecen anuncios", "de lo que hablo"],
    ["Aceptamos términos y condiciones", "sin leer", "vendiendo nuestros datos", "por usar apps gratis"],
    
    # Sobre el futuro
    ["Si la IA hace todo", "¿qué sentido tiene", "estudiar o trabajar?", "crisis existencial 2.0"],
    ["Los trabajos del futuro", "no existen todavía", "y los de ahora", "los está tomando la IA"],
    ["Me preocupa", "que la IA tome decisiones", "sobre mi vida", "sin que yo pueda cuestionarla"],
    ["La IA está en", "medicina, justicia, educación", "¿confiamos tanto", "en la tecnología?"],
]

opiniones_ia = [
    ["la tecnología avanza", "más rápido", "que nuestra ética", "y eso asusta"],
    ["delegamos tanto", "en las máquinas", "que olvidamos", "cómo hacer cosas básicas"],
    ["la eficiencia no es todo", "a veces el proceso humano", "tiene valor", "aunque sea lento"],
    ["nos están vendiendo", "la IA como solución", "pero nadie habla", "de lo que perdemos"],
    ["la automatización", "nos quita trabajos", "pero también", "propósito y significado"],
    ["dependemos de la IA", "para pensar", "para crear", "para decidir. Mal camino"],
    ["cada vez que uso IA", "siento que", "renuncio a algo", "de mi humanidad"],
    ["la IA es útil", "pero estamos", "perdiendo habilidades", "que no recuperaremos"],
]

sentimientos = ["positivo", "negativo", "neutral"]

def generar_fecha():
    inicio = datetime(2022, 1, 1)
    fin = datetime(2024, 12, 31)
    delta = fin - inicio
    dias_random = random.randint(0, delta.days)
    fecha = inicio + timedelta(days=dias_random)
    return fecha.strftime("%d/%m/%Y")

def generar_usuario():
    return f"user_{random.randint(1, 99999)}"

def generar_texto_gen_z():
    patron = random.randint(1, 4)
    
    if patron == 1:
        # Usar una frase completa
        frase = random.choice(frases_gen_z)
        return f"{frase[0]} {frase[1]}, {frase[2]}. {frase[3]}."
    elif patron == 2:
        # Usar opinión
        op = random.choice(opiniones_gen_z)
        return f"{op[0]} {op[1]} {op[2]}. {op[3]}."
    elif patron == 3:
        # Combinar frase + opinión
        frase = random.choice(frases_gen_z)
        op = random.choice(opiniones_gen_z)
        return f"{frase[0]} {frase[1]}. {op[2]} {op[3]}."
    else:
        # Variación de frase
        frase = random.choice(frases_gen_z)
        return f"{frase[0]} {frase[1]}. {frase[2]} {frase[3]}."

def generar_texto_ia():
    patron = random.randint(1, 4)
    
    if patron == 1:
        # Usar una frase completa
        frase = random.choice(frases_ia)
        return f"{frase[0]}, {frase[1]}, {frase[2]}. {frase[3]}."
    elif patron == 2:
        # Usar opinión
        op = random.choice(opiniones_ia)
        return f"{op[0]} {op[1]}, {op[2]} {op[3]}."
    elif patron == 3:
        # Combinar frase + opinión
        frase = random.choice(frases_ia)
        op = random.choice(opiniones_ia)
        return f"{frase[0]} {frase[1]}. {op[2]} {op[3]}."
    else:
        # Variación de frase
        frase = random.choice(frases_ia)
        return f"{frase[0]} {frase[1]}. {frase[2]} {frase[3]}."

def generar_registro(tema_nombre, generar_func):
    fecha = generar_fecha()
    usuario = generar_usuario()
    texto = generar_func()
    tema = tema_nombre
    sentimiento = random.choice(sentimientos)
    likes = random.randint(100, 20000)
    reposts = random.randint(50, 5000)
    
    return [fecha, usuario, texto, tema, sentimiento, likes, reposts]

# Generar dataset
registros = []

print("⏳ Generando 20,000 registros con textos realistas...")

# 10,000 registros de Generación Z
for i in range(10000):
    if i % 2000 == 0:
        print(f"   📱 Generando Gen Z: {i}/10000")
    registros.append(generar_registro("Generación Z y crisis de sentido", generar_texto_gen_z))

# 10,000 registros de IA
for i in range(10000):
    if i % 2000 == 0:
        print(f"   🤖 Generando IA: {i}/10000")
    registros.append(generar_registro("IA y pérdida de autonomía humana", generar_texto_ia))

print("🔀 Mezclando registros...")
random.shuffle(registros)

# Agregar IDs
registros_con_id = [[i+1] + reg for i, reg in enumerate(registros)]

print("💾 Guardando CSV...")
with open('dataset_tweets.csv', 'w', newline='', encoding='utf-8') as archivo:
    escritor = csv.writer(archivo)
    escritor.writerow(['id', 'fecha', 'usuario', 'texto', 'tema', 'sentimiento', 'likes', 'reposts'])
    escritor.writerows(registros_con_id)

print("\n✅ Dataset generado exitosamente!")
print(f"📊 Total de registros: {len(registros_con_id)}")
print(f"📁 Archivo guardado como: dataset_tweets.csv")
print(f"\n📈 Distribución:")
print(f"   - Generación Z y crisis de sentido: 10,000 registros")
print(f"   - IA y pérdida de autonomía humana: 10,000 registros")
print(f"\n💬 Textos realistas que incluyen:")
print(f"   ✓ Burnout y productividad tóxica")
print(f"   ✓ Redes sociales y salud mental")
print(f"   ✓ TikTok, Instagram, algoritmos")
print(f"   ✓ Crisis existencial y económica")
print(f"   ✓ Dependencia de IA")
print(f"   ✓ Privacidad y automatización")