# Checklist de verificación — Proyecto challenge6

## Preparación del entorno

- [ ] **Python**: Tener Python 3.11+ instalado.
- [ ] **Entorno virtual**: Crear y activar un entorno virtual:

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# Windows cmd
.venv\Scripts\activate.bat
```
- [ ] **Actualizar pip**:

```bash
pip install --upgrade pip
```
- [ ] **Instalar dependencias**:

```bash
pip install -r requirements.txt
```

## Datos

- [ ] **Archivos**: Verificar que existan `dataset/psam_husa.csv` y `dataset/psam_husb.csv`.
- [ ] **Formato**: Comprobar tipos de columnas y valores nulos.

## Desarrollo y pruebas

- [ ] **Ejecutar tests**:

```bash
pytest -q
```
- [ ] **Ejecutar CLI de ejemplo** (entrena o lanza flujos definidos):

```bash
challenge6-train
# o
python -m challenge6
```
- [ ] **Generar gráficas**: Ejecutar `generar_graficas.py` para validar visualizaciones.

## Experimentos y reproducción

- [ ] **Correr un experimento de prueba** y confirmar que se guardan métricas en `runs/`.
- [ ] **Registrar versiones**: Si el experimento es reproducible, fijar versiones en `pyproject.toml`/`requirements.txt`.

## Mantenimiento y calidad

- [ ] **Revisar `README.md`** para pasos de uso y ejemplos.
- [ ] **Actualizar `requirements.txt`** tras añadir o cambiar dependencias.
- [ ] **Agregar testing o CI** si es necesario (p.ej., GitHub Actions).
