import numpy as np
import holoviews as hv
from holoviews import streams
import panel as pn

hv.extension('bokeh')
pn.extension()

# Genera datos de ejemplo
data = np.random.rand(100, 100)

# Crea una imagen con los datos de los espectros
image = hv.Image(data).opts(width=400, height=400, tools=['box_select'], colorbar=True, cmap='viridis')

# Define una función para actualizar el gráfico del espectro
def update_spectrum(bounds):
    if not bounds:
        return hv.Curve([])

    x0, y0, x1, y1 = bounds
    ix0, iy0, ix1, iy1 = int(x0), int(y0), int(x1), int(y1)

    sub_image = data[iy0:iy1, ix0:ix1]
    spectrum = np.sum(sub_image, axis=(0, 1))

    return hv.Curve(spectrum)

# Vincula la función de actualización a los eventos de selección de área
bounds_stream = streams.BoundsXY(source=image, bounds=None)
dynamic_spectrum = hv.DynamicMap(update_spectrum, streams=[bounds_stream])

# Establecer opciones de estilo
dynamic_spectrum.opts(height=400, width=400)

# Crea un panel con la imagen y el gráfico del espectro
layout = pn.Row(image, dynamic_spectrum)

# Visualiza el panel
layout.servable()
