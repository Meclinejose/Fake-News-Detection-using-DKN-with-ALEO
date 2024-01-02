import PySimpleGUI as sg
import numpy as np
import matplotlib.pyplot as plt
from Main import Run

sg.change_look_and_feel('DarkBrown3')  # look and feel theme

# Designing layout
layout = [
            [sg.Text("\t\t\tSelect \t   "), sg.Combo(["Learning data(%)","K Value"], size=(13, 20)), sg.Text(""),
           sg.InputText(size=(10, 20), key='1'), sg.Button("START", size=(10, 2))], [sg.Text('\n')],
        [sg.Text(
              "\t\t  ML\t\t\tFakeBERT \t\t  MVAN\t       Ensemble-based DL model\tProposed ALEO-DKN")],
          [sg.Text('Precision    '), sg.In(key='11', size=(20, 20)), sg.In(key='12', size=(20, 20)),
           sg.In(key='13', size=(20, 20)), sg.In(key='14', size=(20, 20)), sg.In(key='15', size=(20, 20))],
          [sg.Text('Recall        '), sg.In(key='21', size=(20, 20)), sg.In(key='22', size=(20, 20)),
           sg.In(key='23', size=(20, 20)), sg.In(key='24', size=(20, 20)), sg.In(key='25', size=(20, 20))],
          [sg.Text('F Measure  '), sg.In(key='31', size=(20, 20)), sg.In(key='32', size=(20, 20)),
           sg.In(key='33', size=(20, 20)), sg.In(key='34', size=(20, 20)), sg.In(key='35', size=(20, 20))],
[sg.Text('Rouge        '), sg.In(key='41', size=(20, 20)), sg.In(key='42', size=(20, 20)),
           sg.In(key='43', size=(20, 20)), sg.In(key='44', size=(20, 20)), sg.In(key='45', size=(20, 20))],
          [sg.Text('\t\t\t\t\t\t\t\t\t\t\t\t            '), sg.Button('Run Graph'), sg.Button('CLOSE')]]


# to plot graphs
def plot_graph(result_1, result_2, result_3,result_4):
    plt.figure(dpi=120)
    loc, result = [], []
    result.append(result_1)  # appending the result
    result.append(result_2)
    result.append(result_3)
    result.append(result_4)

    result = np.transpose(result)

    # labels for bars
    labels = [ 'ML','FakeBERT', 'MVAN',
              'Ensemble-based DL model',
              'Proposed ALEO-DKN']  # x-axis labels ############################
    tick_labels = ['Precision','Recall', 'F Measure','Rouge' ]  #### metrics
    bar_width, s = 0.15, 0  # bar width, space between bars

    for i in range(len(result)):  # allocating location for bars
        if i is 0:  # initial location - 1st result
            tem = []
            for j in range(len(tick_labels)):
                tem.append(j + 1)
            loc.append(tem)
        else:  # location from 2nd result
            tem = []
            for j in range(len(loc[i - 1])):
                tem.append(loc[i - 1][j] + s + bar_width)
            loc.append(tem)

    # plotting a bar chart
    for i in range(len(result)):
        plt.bar(loc[i], result[i], label=labels[i], tick_label=tick_labels, width=bar_width)

    plt.legend(loc=(0.25, 0.25))  # show a legend on the plot -- here legends are metrics
    plt.show()  # to show the plot


# Create the Window layout
window = sg.Window('194313', layout)

# event loop
while True:
    event, values = window.read()  # displays the window
    if event == "START":
        if values[0] == 'Learning data(%)':
            tp = int(values['1']) / 100
        else:
            tp = (int(values['1']) - 1) / int(values['1'])  # k-fold calculation


        print("\n Running..")

        PRE,REC,FM,ROUGE = Run.callmain(tp)

        window.element('11').Update(PRE[0])
        window.element('12').Update(PRE[1])
        window.element('13').Update(PRE[2])
        window.element('14').Update(PRE[3])
        window.element('15').Update(PRE[4])

        window.element('21').Update(REC[0])
        window.element('22').Update(REC[1])
        window.element('23').Update(REC[2])
        window.element('24').Update(REC[3])
        window.element('25').Update(REC[4])

        window.element('31').Update(FM[0])
        window.element('32').Update(FM[1])
        window.element('33').Update(FM[2])
        window.element('34').Update(FM[3])
        window.element('35').Update(FM[4])

        window.element('41').Update(ROUGE[0])
        window.element('42').Update(ROUGE[1])
        window.element('43').Update(ROUGE[2])
        window.element('44').Update(ROUGE[3])
        window.element('45').Update(ROUGE[4])


    if event == 'Run Graph':
        plot_graph(PRE,REC,FM,ROUGE)
    if event == 'CLOSE':
        break
        window.close()
