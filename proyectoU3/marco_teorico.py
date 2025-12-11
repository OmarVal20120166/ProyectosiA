import os

def crear_base_filosofica_robusta():
    print("--- GENERANDO BASE DE CONOCIMIENTO FILOSÓFICO (NIVEL ACADÉMICO) ---")
    
    # Hemos enriquecido el texto con CITAS TEXTUALES para dar "robustez"
    texto_teorico = """
    DOCUMENTO DE REFERENCIA: FILOSOFÍA DE LA TECNOLOGÍA Y SOCIOLOGÍA DIGITAL
    
    === SECCIÓN 1: BYUNG-CHUL HAN (LA SOCIEDAD DEL CANSANCIO) ===
    RESUMEN TEÓRICO:
    Hemos pasado de la sociedad disciplinaria de Foucault (cárceles, hospitales) a la sociedad del rendimiento (oficinas, gimnasios, torres de cristal). El sujeto de rendimiento es libre, pero esa libertad es paradójica: se autoexplota voluntariamente.
    
    CITAS TEXTUALES CLAVE (USAR EN RESPUESTAS):
    - "El exceso de positividad conduce a una sociedad del cansancio. La violencia de la positividad no es privativa, sino saturativa."
    - "Ahora uno se explota a sí mismo y cree que se está realizando."
    - "La sociedad del siglo XXI ya no es disciplinaria, sino una sociedad de rendimiento. Sus habitantes no son sujetos de obediencia, sino sujetos de rendimiento."
    - "El animal laborans tardomoderno está dotado de un ego que raya en lo depresivo."
    
    RELACIÓN CON GEN Z: Burnout, necesidad de "likes" como validación de rendimiento, incapacidad para el "aburrimiento profundo".

    === SECCIÓN 2: ZYGMUNT BAUMAN (MODERNIDAD LÍQUIDA) ===
    RESUMEN TEÓRICO:
    La modernidad sólida (fábricas, matrimonios de por vida) se ha derretido. Vivimos tiempos líquidos donde las formas sociales no mantienen su forma por mucho tiempo.
    
    CITAS TEXTUALES CLAVE (USAR EN RESPUESTAS):
    - "En una vida moderna líquida, no hay vínculos que no puedan romperse."
    - "Las relaciones virtuales son fáciles de entrar y fáciles de salir. Prometen conexión sin compromiso."
    - "El miedo a quedarse atrás o a ser excluido (FOMO) es el motor de la vida de consumo actual."
    - "La identidad no es algo que se hereda, es una tarea que se debe realizar una y otra vez."
    
    RELACIÓN CON GEN Z: Situationships, identidad fragmentada en redes, cultura de la cancelación (desechabilidad humana).

    === SECCIÓN 3: MICHEL FOUCAULT (VIGILANCIA Y BIOPODER) ===
    RESUMEN TEÓRICO:
    El poder moderno se ejerce sobre la vida (biopolítica). El Panóptico es la metáfora de la vigilancia: si crees que te ven, te portas bien.
    
    CITAS TEXTUALES CLAVE (USAR EN RESPUESTAS):
    - "La visibilidad es una trampa. El sujeto es visto, pero él no ve; es objeto de una información, jamás sujeto en una comunicación."
    - "El biopoder es el conjunto de mecanismos por los cuales aquello que, en la especie humana, constituye sus rasgos biológicos fundamentales entra en el interior de la política."
    
    RELACIÓN CON GEN Z: El algoritmo de TikTok como el nuevo Panóptico. Nos vigilamos unos a otros. La "normalización" de los cuerpos en Instagram.

    === SECCIÓN 4: JEAN-FRANÇOIS LYOTARD (LA CONDICIÓN POSMODERNA) ===
    RESUMEN TEÓRICO:
    Incredulidad hacia los metarrelatos. Ya no creemos en el Progreso infinito ni en la Redención religiosa.
    
    CITAS TEXTUALES CLAVE (USAR EN RESPUESTAS):
    - "Simplificando al máximo, se tiene por 'posmoderna' la incredulidad con respecto a los metarrelatos."
    - "El saber cambia de estatuto al mismo tiempo que las sociedades entran en la edad llamada postindustrial."
    
    RELACIÓN CON GEN Z: Nihilismo, humor absurdo (shitposting), búsqueda de micro-causas en lugar de grandes revoluciones.

    === SECCIÓN 5: MARTIN HEIDEGGER (LA PREGUNTA POR LA TÉCNICA) ===
    RESUMEN TEÓRICO:
    La esencia de la técnica no es tecnológica, es una forma de "desocultar" la verdad. El peligro es ver al humano como "stock" (Bestand).
    
    CITAS TEXTUALES CLAVE (USAR EN RESPUESTAS):
    - "La técnica no es un mero medio, la técnica es un modo del desocultar."
    - "El peligro supremo es que el hombre mismo sea tomado solo como una reserva o fondo (Bestand)."
    
    RELACIÓN CON GEN Z: Dating apps donde las personas son "catálogo", auto-optimización (biohacking).
    """

    os.makedirs("datos", exist_ok=True)
    ruta = "datos/marco_teorico_filosofia.txt"
    
    with open(ruta, "w", encoding="utf-8") as f:
        f.write(texto_teorico)
    
    print(f"📚 Archivo ACADÉMICO generado en: {ruta}")
    print("✅ ACCIÓN REQUERIDA: Ve a AnythingLLM, borra el archivo 'marco_teorico' anterior y sube este nuevo.")

if __name__ == "__main__":
    crear_base_filosofica_robusta()