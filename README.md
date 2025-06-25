# Predict Gender based on name

## ⚙️ Opciones Avanzadas

### Usar un modelo personalizado

```python
genderizer = LatamGenderize(model_path='ruta/a/tu/modelo.h5')
```

### Especificar la columna de nombres

```python
result = genderizer.genderize(df, name_column='primer_nombre')
```

---

## 📁 Estructura del Paquete

```
latam_genderize/
├── __init__.py
├── genderize.py
└── models/
    ├── boyorgirl_CO_ES.h5   # Modelo por defecto (no incluido aquí)
    └── README.md
```

---

## 🧪 Tests

Incluye tests unitarios en `tests/`. Para ejecutarlos:

```bash
pip install -e .[dev]
pytest
```

---

## 🛠️ Entrenamiento y formato del modelo

- El modelo debe ser un archivo `.h5` de Keras/TensorFlow.
- Debe aceptar como input un array de nombres codificados a nivel de caracter (ver código fuente para detalles).
- Puedes reemplazar el modelo por defecto por uno propio, siempre que respete el formato de entrada.

---

## 📚 Ejemplo completo

Consulta el archivo [`examples/basic_usage.py`](examples/basic_usage.py) para un ejemplo de uso más detallado.

---

## ❓ Preguntas frecuentes

- **¿Qué pasa si no tengo la columna 'name'?**  
  El paquete intenta detectar automáticamente columnas como `nombre`, `first_name`, etc. Si no la encuentra, puedes especificarla manualmente.

- **¿Puedo usar mi propio modelo?**  
  Sí, solo pásalo como argumento a `LatamGenderize(model_path=...)`.

- **¿Qué precisión tiene el modelo?**  
  Depende del modelo `.h5` que uses. El paquete solo provee la infraestructura.

---

## 📄 Licencia

MIT. Consulta el archivo `LICENSE`.

---

## 👤 Autor

Desarrollado por [Tu Nombre]  
Contacto: [tu.email@ejemplo.com]

---

**¡Contribuciones y sugerencias son bienvenidas!** 