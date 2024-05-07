#Soriano Rios Sergio Miguel - 20210642
#Modelo Predictivo Para Enfermedades

# Importación de los módulos necesarios
import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
from tkinter import filedialog
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd

# Clase para el modelo de detección temprana de enfermedades
class EarlyDetectionModel:
    def __init__(self):
        self.model = DecisionTreeClassifier()

    # Método para entrenar el modelo
    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    # Método para hacer predicciones
    def predict(self, X_test):
        return self.model.predict(X_test)

# Clase principal de la aplicación
class Application(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Modelo Predictivo De Enfermedades")
        self.geometry("500x250")

        self.model = EarlyDetectionModel()  # Inicialización del modelo
        self.pacientes = []  # Lista para almacenar datos de pacientes
        self.enfermedades = {}  # Diccionario para almacenar enfermedades y síntomas
        self.model_trained = False  # Bandera para indicar si el modelo ha sido entrenado
        self.mlb = None  # Objeto MultiLabelBinarizer

        self.create_widgets()  # Creación de los widgets de la interfaz

    # Método para crear los widgets de la interfaz
    def create_widgets(self):
        self.tabControl = ttk.Notebook(self)

        self.tab1 = ttk.Frame(self.tabControl)
        self.tab2 = ttk.Frame(self.tabControl)
        self.tab3 = ttk.Frame(self.tabControl)

        self.tabControl.add(self.tab1, text="Alta Paciente")
        self.tabControl.add(self.tab2, text="Pacientes Dados de Alta")
        self.tabControl.add(self.tab3, text="Modelo Predictivo")

        self.tabControl.pack(expand=1, fill="both")

        # Tab 1: Alta Paciente
        self.lbl_nombre = ttk.Label(self.tab1, text="Nombre del paciente:")
        self.lbl_nombre.grid(row=0, column=0, padx=10, pady=10)
        self.nombre_entry = ttk.Entry(self.tab1)
        self.nombre_entry.grid(row=0, column=1, padx=10, pady=10)

        self.lbl_edad = ttk.Label(self.tab1, text="Edad del paciente:")
        self.lbl_edad.grid(row=1, column=0, padx=10, pady=10)
        self.edad_entry = ttk.Entry(self.tab1)
        self.edad_entry.grid(row=1, column=1, padx=10, pady=10)

        self.lbl_sintomas = ttk.Label(self.tab1, text="Síntomas del paciente:")
        self.lbl_sintomas.grid(row=2, column=0, padx=10, pady=10)
        self.sintomas_entry = ttk.Entry(self.tab1)
        self.sintomas_entry.grid(row=2, column=1, padx=10, pady=10)

        self.btn_alta = ttk.Button(self.tab1, text="Dar de alta", command=self.dar_de_alta)
        self.btn_alta.grid(row=3, column=1, padx=10, pady=10)

        # Tab 2: Pacientes Dados de Alta
        self.pacientes_listbox = tk.Listbox(self.tab2)
        self.pacientes_listbox.pack(fill=tk.BOTH, expand=True)

        # Tab 3: Modelo Predictivo
        self.lbl_seleccion_paciente = ttk.Label(self.tab3, text="Seleccione un paciente:")
        self.lbl_seleccion_paciente.pack(pady=10)
        self.pacientes_combobox = ttk.Combobox(self.tab3, values=[], state="readonly")
        self.pacientes_combobox.pack(pady=10)

        self.btn_cargar_datos = ttk.Button(self.tab3, text="Cargar Datos de Entrenamiento", command=self.cargar_datos_entrenamiento)
        self.btn_cargar_datos.pack(pady=5)

        self.btn_entrenar_modelo = ttk.Button(self.tab3, text="Entrenar Modelo", command=self.entrenar_modelo)
        self.btn_entrenar_modelo.pack(pady=5)

        self.btn_prediccion = ttk.Button(self.tab3, text="Predecir enfermedad", command=self.predecir_enfermedad)
        self.btn_prediccion.pack(pady=10)

    # Método para dar de alta a un paciente
    def dar_de_alta(self):
        nombre = self.nombre_entry.get()
        edad = self.edad_entry.get()
        sintomas = self.sintomas_entry.get().split(",")
        paciente = {'Nombre': nombre, 'Edad': edad, 'Síntomas': sintomas}
        self.pacientes.append(paciente)  # Agrega paciente a la lista
        self.pacientes_listbox.insert(tk.END, f"Nombre: {nombre}, Edad: {edad}, Síntomas: {sintomas}")  # Actualiza la lista visual
        self.actualizar_combobox_pacientes()  # Actualiza el combobox de pacientes
        self.limpiar_campos_texto()  # Limpia los campos de texto
        messagebox.showinfo("Paciente Agregado", "El paciente se ha agregado correctamente.")

    # Método para cargar los datos de entrenamiento desde un archivo
    def cargar_datos_entrenamiento(self):
        archivo = filedialog.askopenfilename(title="Seleccione el archivo de datos de entrenamiento", filetypes=[("Excel files", "*.xlsx")])
        if archivo:
            try:
                self.enfermedades = self.cargar_enfermedades(archivo)
                messagebox.showinfo("Datos Cargados", "Los datos de entrenamiento se han cargado correctamente.")
            except Exception as e:
                messagebox.showerror("Error", f"Error al cargar los datos: {str(e)}")

    # Método para entrenar el modelo
    def entrenar_modelo(self):
        if not self.enfermedades:
            messagebox.showwarning("Advertencia", "Debe cargar los datos de entrenamiento antes de entrenar el modelo.")
            return

        X_train = []  # Lista para almacenar los síntomas de los pacientes
        y_train = []  # Lista para almacenar las enfermedades de los pacientes

        for enfermedad, sintomas in self.enfermedades.items():
            for paciente in self.pacientes:
                if any(sintoma in paciente['Síntomas'] for sintoma in sintomas):
                    X_train.append(paciente['Síntomas'])  # Pasar los síntomas como lista de cadenas de texto
                    y_train.append(enfermedad)

        if X_train and y_train:
            mlb = MultiLabelBinarizer()  # Inicializar el codificador one-hot
            X_train_encoded = mlb.fit_transform(X_train)  # Codificar los síntomas
            self.model.train(X_train_encoded, y_train)
            self.mlb = mlb  # Guardar el MultiLabelBinarizer ajustado
            self.model_trained = True  # Marcar el modelo como entrenado
            messagebox.showinfo("Modelo Entrenado", "El modelo ha sido entrenado con éxito.")
        else:
            messagebox.showwarning("Sin Datos", "No hay suficientes datos para entrenar el modelo.")

    # Método para predecir la enfermedad de un paciente
    def predecir_enfermedad(self):
        if len(self.pacientes) < 1:
            messagebox.showwarning("Advertencia", "Debe dar de alta al menos un paciente antes de usar el modelo predictivo.")
            return

        paciente_index = self.pacientes_combobox.current()
        paciente = self.pacientes[paciente_index]
        sintomas_paciente = paciente['Síntomas']

        if not self.model_trained:
            messagebox.showwarning("Advertencia", "Debe entrenar el modelo antes de hacer predicciones.")
            return

        mlb = self.mlb  # Se utiliza el MultiLabelBinarizer ya ajustado
        sintomas_paciente_encoded = mlb.transform([sintomas_paciente])  # Codificar los síntomas del paciente
        enfermedad_predicha = self.model.predict(sintomas_paciente_encoded)

        if enfermedad_predicha:
            messagebox.showinfo("Enfermedad Predicha", f"Enfermedad predicha para el paciente: {enfermedad_predicha[0]}")
        else:
            messagebox.showwarning("Sin Predicción", "No se pudo predecir una enfermedad para el paciente.")

    # Método para cargar los datos de enfermedades desde un archivo
    def cargar_enfermedades(self, archivo):
        df = pd.read_excel(archivo)
        enfermedades = {}
        for index, row in df.iterrows():
            enfermedad = row['Enfermedad']
            sintomas = [sintoma.strip() for sintoma in row['Síntomas'].split(",")]
            enfermedades[enfermedad] = sintomas
        return enfermedades

    # Método para actualizar el combobox de pacientes
    def actualizar_combobox_pacientes(self):
        self.pacientes_combobox['values'] = [paciente['Nombre'] for paciente in self.pacientes]

    # Método para limpiar los campos de texto
    def limpiar_campos_texto(self):
        self.nombre_entry.delete(0, tk.END)
        self.edad_entry.delete(0, tk.END)
        self.sintomas_entry.delete(0, tk.END)

# Bloque principal de ejecución
if __name__ == "__main__":
    app = Application()
    app.mainloop()
