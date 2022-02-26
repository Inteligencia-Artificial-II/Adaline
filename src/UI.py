from tkinter import Tk, Frame, Label, Entry, Button, DISABLED, Toplevel, ttk, CENTER, NO
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

def render_conv(self):
    """Definimos la ventana donde se imprime la gráfica de convergencia"""
    # Crea una nueva ventana hija de self.window
    self.conv_window = Toplevel(self.window)
    self.conv_window.title('Error acumulado por época')
    self.conv_window.geometry("650x650")
    Label(self.conv_window, text='Error acumulado por época', font=('Arial', 20)).grid(row=0, columnspan=4)
    FigureCanvasTkAgg(self.conv_fig, self.conv_window).get_tk_widget().grid(row=1, columnspan=4)

def render_confusion_matrix(self):
    """Definimos la ventana donde se imprime la gráfica de convergencia"""
    # Crea una nueva ventana hija de self.window
    self.conf_window = Toplevel(self.window)
    self.conf_window.title('Matriz de confusión')
    self.conf_window.geometry("805x250")
    Label(self.conf_window, text='Matriz de confusión', font=('Arial', 20)).grid(row=0, columnspan=4)
    self.table_frame = Frame(self.conf_window)
    self.table = ttk.Treeview(self.table_frame)
    self.table['columns'] = ('n','Predicciones clase 0', 'Predicciones clase 1', 'Sumatoria')
    self.table.column("#0", width=0,  stretch=NO, anchor=CENTER)

    for i in range(4):
        self.table.column(i, anchor=CENTER)   

    self.table.heading("#0",text="", anchor=CENTER)
    for i in self.table['columns']:
        self.table.heading(i, text=i, anchor=CENTER)

    self.table_frame.grid(rowspan=4, columnspan=4)
    self.table.grid(row=0)
    
def render_gui(self):
    """Definimos la interfaz gráfica de usuario"""
    self.window.title('Adaline')
    self.window.geometry("650x650")

    Label(self.window, text="Adaline", font=("Arial", 20)).grid(row=0, columnspan=4)
    # añade el gráfico de matplotlib a la interfaz de tkinter
    FigureCanvasTkAgg(self.fig, self.window).get_tk_widget().grid(row=1, columnspan=4)

    # contendrá un segmento de la interfaz (valores para entrenar)
    self.container_before = Frame(self.window)

    Label(self.container_before, text="Epocas máximas:").grid(row=0, column=0)
    self.max_iter = Entry(self.container_before)

    Label(self.container_before, text="Tasa de aprendizaje:").grid(row=1, column=0)
    self.learning_rate = Entry(self.container_before)

    Label(self.container_before, text="Error mínimo deseado:").grid(row=2, column=0)
    self.min_error = Entry(self.container_before)

    self.weight_btn = Button(self.container_before, text="Inicializar pesos", command=self.init_weights, state=DISABLED)
    self.run_btn = Button(self.container_before, text="Entrenar", command=self.run, state=DISABLED)

    self.container_before.grid(row=2, columnspan=4)

    self.container_after = Frame(self.window)
    self.analyse = Button(self.container_after, text="Evaluar", command=self.evaluate, state=DISABLED)
    self.restart_btn = Button(self.container_after, text="Reiniciar", command=self.restart)
    self.is_converge = Label(self.container_after, text="", font=("Arial", 15))

    self.max_iter.grid(row=0, column=1)
    self.learning_rate.grid(row=1, column=1)
    self.min_error.grid(row=2, column=1)
    self.weight_btn.grid(row=0, column=2)
    self.run_btn.grid(row=1, column=2)
    self.analyse.grid(row=0, column=0)
    self.restart_btn.grid(row=0, column=1)
    self.is_converge.grid(row=1, columnspan=3)
    # escucha los eventos del mouse sobre el gráfico
    self.fig.canvas.mpl_connect('button_press_event', self.set_point)

    # termina el programa al hacer click en la X roja de la ventana
    self.window.protocol('WM_DELETE_WINDOW', lambda: quit())
