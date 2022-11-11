from PyQt5.QtWidgets import QApplication, QMainWindow
import sys
import index1
import data1,algorithm1,result1

# def show_sub():
#     MainWindow1 = QMainWindow()
#     index_ui = data1.Ui_Dialog()
#     index_ui.gonext.clicked.co
#     index_ui.setupUi(MainWindow1)
#     MainWindow1.show()


def show_data():
    MainWindow1.hide()
    MainWindow2.show()
def back_data():
    MainWindow3.hide()
    MainWindow2.show()
def back_index():
    MainWindow2.hide()
    MainWindow1.show()
def show_algorithm():
    MainWindow2.hide()
    MainWindow3.show()
def back_algorithm():
    MainWindow4.hide()
    MainWindow3.show()
def show_result():
    MainWindow3.hide()
    # MainWindow4.close()
    # result_ui.setupUi(MainWindow4)
    MainWindow4.show()
def data_val(val):
    data_ui.signal2.emit(val)
def algorithm_val(val):
    algorithm_ui.signal2.emit(val)
def result_val(val):
    result_ui.signal2.emit(val)



if __name__ == '__main__':
    app = QApplication(sys.argv)
    MainWindow1 = QMainWindow()
    index_ui = index1.Ui_Dialog()
    index_ui.setupUi(MainWindow1)

    MainWindow2 = QMainWindow()
    data_ui = data1.Ui_Dialog()
    data_ui.setupUi(MainWindow2)

    MainWindow3 = QMainWindow()
    algorithm_ui = algorithm1.Ui_Dialog()
    algorithm_ui.setupUi(MainWindow3)

    MainWindow4 = QMainWindow()
    result_ui = result1.Ui_Dialog()
    result_ui.setupUi(MainWindow4)

    index_ui.gonext.clicked.connect(show_data)
    index_ui.signal1.connect(data_val)
    data_ui.openfile.clicked.connect(back_index)
    data_ui.openfile_2.clicked.connect(show_algorithm)
    data_ui.signal1.connect(algorithm_val)
    algorithm_ui.openfile.clicked.connect(back_data)
    algorithm_ui.pushButton_2.clicked.connect(show_result)
    algorithm_ui.pushButton_5.clicked.connect(show_result)
    algorithm_ui.pushButton_6.clicked.connect(show_result)
    algorithm_ui.pushButton_7.clicked.connect(show_result)
    algorithm_ui.signal1.connect(result_val)
    result_ui.openfile.clicked.connect(back_algorithm)
    MainWindow1.show()
    # the_mainwindow = index1.Ui_Dialog()
    # the_mainwindow.show()
    sys.exit(app.exec_())

