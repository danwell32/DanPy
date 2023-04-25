#Importar las librerias: 
import hyperspy.api as hs
import numpy as np
import h5py


def read_hdf5_TsXM(file_name):
    """
    Lee un archivo hdf5 que contiene los datos de TsXM.
    Parameters:
    -----------
    file_name: string.
        Nombre del archivo hdf5 que se desea leer.

    Returns:
    --------
    metadata: dict.
        Diccionario con las claves contenidas en el archivo hdf5. La clave es una cadena de texto con el formato 'índice.nombre_clave' y el valor es la forma del dato correspondiente.
    data: list.
        Lista de los datos contenidos en el hdf5.
    """

    metadata = {}
    data = []
    with h5py.File(file_name, 'r') as f:
        initial = list(f.keys())
        items = f[initial[0]]
        for i, key in enumerate(items):
            shape = f['FastAligned'][key].shape
            metadata[f"{i}.{key}"] = shape
            data.append(f['FastAligned'][key][:])
    return metadata, data

def spectrum_to_array(data,index):
    """
    Función para extraer los datos en formato numpy array y calcular el logaritmo negativo de los mismos.
    Parameters:
    -----------
    data: lista.
        Datos obtenidos de la función "read_hdf5_TsXM".
    index: int.
        Índice de la señal de espectros alineados.
        
    Returns:
    --------
    matrix: numpy array.
        Señal de espectros en formato 2D de hyperspy. 
    """
    matriz = np.moveaxis(data[index],0,2)
    matriz[matriz<=0] = 1 #Eliminamos todo valor negativo o zero para evitar que el logaritmo nos de error. 
    matrix = -np.log(matriz)
    return matrix


def spectrum_to_hs(data, index):
    """
    Función para generar una imagen de espectros en formato de hyperspy, y aplicar el menos logaritmo natural en los datos. 
    
    Parameters:
    -----------
    data: list. Lista de datos obtenida de la función `read_hdf5_TsXM`.
    index: int. Indice correspondiente a la imagen de espectros que se desea procesar.
    
    Returns:
    --------
    signal: objeto de tipo `hs.signals.Signal2D`. Imagen de espectros en formato de hyperspy, con las siguientes propiedades definidas:
        - signal.axes_manager[0].name: 'x'
        - signal.axes_manager[1].name: 'y'
        - signal.axes_manager[2].name: 'energy'
        - signal.axes_manager[-1].offset: valor de offset para el eje de energía
        - signal.axes_manager[-1].scale: valor de escala para el eje de energía
        - signal.axes_manager[0].scale: valor de escala para el eje x
        - signal.axes_manager[1].scale: valor de escala para el eje y
    """
    matriz = np.moveaxis(data[index], 0, 2)
    matriz[matriz <= 0] = 1
    matrix = -np.log(matriz)
    signal = hs.signals.Signal2D(matrix)
    signal.axes_manager.set_signal_dimension(1)
    signal.axes_manager[0].name = 'x'
    signal.axes_manager[1].name = 'y'
    signal.axes_manager[2].name = 'energy'
    signal.axes_manager[-1].offset = data[2][0]
    signal.axes_manager[-1].scale = data[2][1] - data[2][0]
    signal.axes_manager[0].scale = data[6][:]
    signal.axes_manager[1].scale = data[7][:]
    return signal

