# Fake-News-Detection-using-DKN-with-ALEO
Adaptive lotus effect optimization with deep Kronecker network for fake news detection on social media with Tamil language

**1.	PROJECT TITLE:**
Adaptive lotus effect optimization with deep Kronecker network for fake news detection on social media with Tamil language

**2.	HARDWARE REQUIREMENTS**
	OS-Windows 10 or MacOS 
	RAM-8GB
	ROM-More than 100 GB
	GPU-Yes
	CPU-1.7 GHz

**3.	SOFTWARE REQUIREMENTS**
	Software name(Python): Version: 3.9.11
	(Download link: https://www.python.org/downloads/release/python-376/ )
		Click -> Windows x86-64 executable installer.
 
	Software name: PyCharm: Version: 2020.3.3 
	(Download link: https://www.jetbrains.com/pycharm/download/other.html)
 
	(For installation procedure, please refer the doc “steps to install python.doc”)

**4. HOW TO RUN**
Step 1: Loading the project in PYCHARM
Open pycharm
Go to File, select Open browse the project from your drive and select it. So that the project will get loaded into the Pycharm.<br /> 
For the first time, Pycharm will take some time to load the settings.<br /> 
Please wait if any process is loading on the bottom of the screen.<br /> 
Check the Project Interpreter (File -> Settings -> Project: 194313-> Project Interpreter). <br /> 
If this location “(C:\Users\---\AppData\Local\Programs\Python\Python39-64\python.exe) is not presented, then add this ‘python.exe’ from the installed location.<br /> 
In Pycharm Terminal(bottom left), type the comment “pip install -r requirements.txt”<br /> 
<br /> 
<br /> 
Step 2: Run the program and getting the results <br /> 
From 'current project folder' window in pycharm, Open 194313-> Main->GUI.py’ and click run button<br /> 
In GUI window,<br /> 
1) Enter  Learning data(%)(eg:60,70,80,90) or K Value (eg:5,6,7,8)<br /> 
 2) Click START, after some time the result will be displayed <br /> 
 3) Click Run Graph to view the current result graph.<br /> 
[Expected Execution time expected: 15 – 20 minutes]
<br /> 
Step 3: Generate the graphs plotted in the paper<br />
From 'current project folder' window in pycharm, open ‘194313-> Main->Result_graphs.py’, and click run button.<br /> 
<br /> 
<br /> 
**5. IMPORTANT PYTHON FILE AND DESCRIPTION:**
Main-> GUI.py: User Interface, code starts here
Main-> Run.py: Main code
Main->Bert_Tokenization.py :  tokenization using BERT
Main->Fea_Ext.py  : feature extraction (Word2Vec, no. of numerical values, ,hashtags, punctuation marks , Numerical words , lin similarity score TF-IDF based features,)
Main->Proposed_ALEO_DKN ->DKN.py ,ALEA.py: fake news detection using DKN with adaptive lotus effect optimization  
Main-> Result_graphs.py: displays graphs in paper.

