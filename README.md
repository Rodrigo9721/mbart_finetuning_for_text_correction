# mbart_finetuning_for_text_correction
Este ejercicio busca realizar el fintuning de un transformer mBart para la correción de texto en español.
Para esto, se necesita una base de datos de texto corregido apropiadamente. Para esto, se realizó lo siguiente:
- Extracción de 30k artículos de wikipedia
- Separación de todas las frases de cada artículo
- Se agregaron artificialmente errores en estos artículos para así crear la base de datos

Comentarios:
- El modelo mBart large es un transformer que demanda muchos recursos de computacióm, así que aún no he podido probar el finetuning.
- Se probó con el modelo t5-small, pero los resultados no fueron buenos por el hecho de que estaba entrenado principalmente en inglés (mbart tiene 51 idiomas)
