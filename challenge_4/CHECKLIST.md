# Checklist - Challenge 4 (Grupo 6: Venture)

## 1. Comandos Exactos de Ejecución
Para reproducir el pipeline completo desde la raíz del proyecto, ejecutar en orden:
* **Recolección de Demostraciones Humanas:** `python jugarVenture.py`
* **Entrenamiento Clonación de Comportamiento (Warm-start):** `python train_bc.py`
* **Entrenamiento GAIL (Híbrido):** `python train_gail.py --episodes 10000 --save-prefix gail_long13`
* **Generación de Gráficas de Evaluación:** `python plot_metrics.py --csv gail_long13_metrics.csv`

## 2. Semillas Utilizadas (Reproducibilidad)
* **Entorno, recolección y entrenamiento:** Se utilizó la semilla base `42` (con incrementos `42 + ep` por episodio durante el entrenamiento para garantizar la diversidad del espacio de estados y estocasticidad en la aparición de monstruos).

## 3. Punteros a Registros y Figuras
* **Modelos Entrenados:** `gail_long13_generator.pt` (Actor-Crítico), `gail_long13_discriminator.pt` y `bc_policy.pt`.
* **Registro de Datos Crudos:** `gail_long13_metrics.csv`.
* **Gráficas (Curvas de Aprendizaje y Discriminador):** `gail_long13_metrics_plots.png` (Archivo principal para el análisis de varianza y desequilibrio minimax).
* **Metadatos del Experto:** `demos_info.txt`.

## 4. Resumen Comparativo 
En el entorno *Venture*, caracterizado por penalizaciones severas y recompensas de tesoros extremadamente raras, los agentes RL tradicionales (DQN y PPO) fracasan por inanición de gradientes o parálisis táctica. Para superar esto, implementamos GAIL utilizando demostraciones humanas, descartando los modelos subóptimos de DQN. GAIL demostró un valor inicial muy superior al de DQN y PPO al aprender a navegar fluidamente por las complejas mazmorras y evadir monstruos sin depender de un puntaje, guiado únicamente por la medida de ocupación del experto. 

Sin embargo, nuestro análisis empírico reveló que GAIL es altamente vulnerable a la Confusión Causal y al Sesgo de Supervivencia. El discriminador dominó el juego minimax (precisión $>0.8$), sobreajustándose a las características visuales del humano, lo que generó gradientes ruidosos y una alta varianza en la política. Esto indujo el fenómeno del "Fantasma Errante": el agente imitó a la perfección la mecánica de movimiento (el estilo), pero falló en internalizar el objetivo de recolectar tesoros. Concluimos que GAIL añade un valor inmenso para desbloquear la exploración espacial temprana frente a recompensas raras, pero en dominios complejos requiere arquitecturas condicionadas a metas para convertir la imitación de movimiento en comportamiento óptimo.