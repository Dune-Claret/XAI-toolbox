# TOOL BOX
# The best for understanding.
# 
# TOOL BOX permet de générer des explications de techniques de machine learning. 
# 
# Destiné à recevoir des données tabulaires ou des images et des modèles parmi Random Forest, Neural Network, Regression linéaire et XGBoost, TOOL BOX vous propose une interprétabilité de vos résultats.
# 
# Nécessite l'installation de tkinter, pdpbox, eli5, shap, lime.
##

from tkinter import *
import tkinter as tk
import tkinter.filedialog
import copy
import numpy as np
import pandas as pd
import time
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg 
import matplotlib.pyplot as plt 
import matplotlib 
import math
from scipy.stats.mstats_basic import scoreatpercentile
import os 
import pickle
import lime
import lime.lime_tabular
import eli5
from pdpbox import pdp
from eli5.sklearn import PermutationImportance
import shap

# typographie et taille du texte
LARGE_FONT= ("Times", 40)
LARGE_FONT2=("Times",60)
LARGE_FONT3=("Times",40)
TITLE=("Times",60)
TITLE2=("Times",30)
SUBTITLE=("Times",30)
TEXT=("Times",26)
TEXT_Help=("Times",22)
TEXT_Assistance=("Times",16)
TEXT_Fin=("Times",16)

BUTTON=("Times",20)
Browse_BUTTON=("Times",15)
BigBUTTON=("Times",30)
SUB_BUTTON=("Times",15)
ENTER=("Times",10)

frame_color = 'white' #'wheat1'
text_title_color = 'SkyBlue4'
text_color = 'black'
text_button = 'white'

Big_button_color = 'tomato'
Browse_BUTTON_ft_color = 'SkyBlue4'
Browse_BUTTON_bg_color = 'SlateGray3'
bg_button = 'LightSalmon2'

text_explain="TOOL BOX permet de générer des  explications de techniques de machine learning.\n\n Destiné à recevoir des données tabulaires ou des images et des modèles parmi Random Forest,\n Neural Network, Regression linéaire et XGBoost, TOOL BOX vous propose une interprétabilité \n de vos résultats.\n\n\n\n Nécessite l'installation de tkinter, pdpbox, eli5, shap, lime."
text_TabularPage="Sélectionnez une technique parmi les options suivantes."
text_lime = "veuillez mettre :"
text_assistance = "En savoir plus sur le machine learning interprétable.\nLe développement et l’engouement rapides autour de l’IA ont conduit à prioriser la performance des algorithmes, \nalors qu’à présent la confluence de considérations réglementaires, éthiques, économiques et  fonctionnelles font émerger une nouvelle \ndynamique  dans laquelle l'interprétabilité pourrait devenir le nouveau critère d’évaluation des modèles. \nCette considération nouvelle a conduit au développement d’un secteur de recherche porté sur le machine \nlearning interprétable. L’interprétabilité explicite donc la façon dont la décision a été prise par l’algorithme (réponse au 'comment' ?).\n \nChoisissez la bonne technique d’interprétabilité.\nPDP  montre l'effet marginal qu'une ou deux features ont sur le résultat prévu d'un modèle d'apprentissage machine. \nICE visualise la dépendance de la prédiction à une feature pour chaque instance séparément.\nLIME va regarder l’incidence des variations des features dans le modèle d’apprentissage automatique sur les prédictions.\nSHAP montre l’évolution de la prédiction à chaque ajout de variable.\nPermutation feature permutation permet de mesurer l’impact de chaque Feature sur les prédictions.\n \nOffres de la TOOL BOX \nAfficher et sauvegarder les résultats d’une technique d’interprétabilté choisie sur votre modèle entrainé, votre dataset et vos données \n (entrainement et test) aux formats demandés. Les résultats sont sauvegardés en .png dans votre répertoire courant. \n \nExtensions acceptées pour vos fichiers \nModèles  .sav .pkl \nDataset et données  .csv .pkl .hdf5 .h5 .npy"
Chemin_Image = "/Users/moi/Desktop/marjo/toolbox.png"

global hide
global currentGraph
global currentdirectory
global click


currentdirectory = os.getcwd() 










class XAI_Project(tk.Tk):
    def __init__(self, *args,**kwargs):
        tk.Tk.__init__(self, *args,**kwargs)
        container=tk.Frame(self)
        container.pack(side="top",fill="both",expand=True)
        container.grid_rowconfigure(0,weight=1)
        container.grid_columnconfigure(0, weight=1)
        self.frames={}

        for F in (StartPage,StartPage2,Assistance,OptionPage,TabularPage,LIMETab,PDP,SHAP,FeatureImportance,PermutationImportance, ExplanationPage,PageQuit):
            frame=F(container,self)
            self.frames[F]=frame
            frame.grid(row=0,column=0, sticky="nsew")
        self.show_frame(StartPage)

    def show_frame(self,cont):
        frame=self.frames[cont]
        frame.configure(bg=frame_color) #couleur de fond de la fenetre
        frame.tkraise()










class StartPage (tk.Frame): # ecran d'accueil

    def __init__(self,parent,controller):

        tk.Frame.__init__(self,parent)
        
        global currentdirectory
        image1 = tk.PhotoImage(file=currentdirectory + "/BG.png")
        w = image1.width()
        h = image1.height()
        
        canvas = tk.Canvas(self, width=w, height=h)
        
        
        panel1 = tk.Label(canvas, image=image1)
        panel1.place(relx=0.5, rely=0.5, anchor='center')
        panel1.image = image1
        
        BB=tk.Button(canvas, text="Découvrir",font=TITLE2, command=lambda:controller.show_frame(StartPage2))
        BB.config(fg='white') #couleur du texte
        BB.config(background='red') #couleur de fond du bouton
        
        BB.place(relx=0.5,rely=0.9,anchor='center')
        canvas.pack()









class StartPage2 (tk.Frame): # ecran d'accueil

    def __init__(self,parent,controller):

        tk.Frame.__init__(self,parent)
        # conteneur du titre
        fen_Title = tk.Frame(self, width=1000, height=200,)
        fen_Title.configure(bg=frame_color)#frame_color) #couleur de fond de la fenetre

        global currentdirectory

        label=tk.Label(fen_Title,text="TOOL BOX", font=TITLE)
        label.config(fg='red') #couleur du texte
        label.config(background=frame_color)#frame_color) #couleur de fond du texte
        label.pack(pady = 10) # abaisser le titre
        
        label=tk.Label(fen_Title,text="Your next model is not a black box.", font=SUBTITLE)
        label.config(fg='tomato') #couleur du texte
        label.config(background=frame_color)#frame_color) #couleur de fond du texte
        label.pack(pady = 10) # abaisser le titre

        photo = tk.PhotoImage(file=currentdirectory + "/toolbox.png")
        labelimg = tk.Label(fen_Title, image=photo)
        labelimg.image = photo
        labelimg.pack(pady=70, padx=0)

        # conteneur des boutons
        fen_boutons = tk.Frame(self, width=1000, height=200,)
        fen_boutons.configure(bg=frame_color)#frame_color) #couleur de fond de la fenetre
        
        button1=tk.Button(fen_boutons,text="Continuer",font=BUTTON ,command=lambda:controller.show_frame(ExplanationPage))
        button1.config(fg=text_button) #couleur du texte
        button1.config(background=bg_button) #couleur de fond 
        button1.place(relx=0,rely=1,anchor=tk.SW)

        button2=tk.Button(fen_boutons,text="Assistance",font=BUTTON ,command=lambda:controller.show_frame(Assistance))
        button2.config(fg=text_button) #couleur du texte
        button2.config(background=bg_button) #couleur de fond du texte
        button2.place(relx=0.5,rely=1,anchor=tk.S)

        button3=tk.Button(fen_boutons,text="Quitter",font=BUTTON ,command=lambda:controller.show_frame(PageQuit))
        button3.config(fg=text_button) #couleur du texte
        button3.config(background=bg_button) #couleur de fond du texte
        button3.place(relx=1,rely=1,anchor=tk.SE)
        
        #labelimg.pack(pady=0, padx=0)

        fen_Title.pack(pady=50)
        fen_boutons.pack(pady=20)










class OptionPage (tk.Frame): # Choix du type de donnees a interpreter
    
    def __init__(self,parent,controller,):
        
        tk.Frame.__init__(self,parent)
        # conteneur du titre
        fen_Text = tk.Frame(self, width=1000, height=500,)
        fen_Text.configure(bg=frame_color)#frame_color) #couleur de fond de la fenetre
        
        global currentdirectory
        
        label=tk.Label(fen_Text,text="Données à interpréter \t ", font=LARGE_FONT)
        label.config(fg=text_color) #couleur du texte
        label.config(background=frame_color)#frame_color) #couleur de fond du texte
        label.grid(row=0, column = 0)#pack(pady = 40) # abaisser le titre

        photo = tk.PhotoImage(file=currentdirectory + "/document.png")
        labelimg = tk.Label(fen_Text, image=photo)
        labelimg.image = photo
        labelimg.grid(row=0, column=1)#pack(pady=150, padx=0)

        # conteneur des boutons
        fen_boutons = tk.Frame(self, width=1000, height=600,)
        fen_boutons.configure(bg=frame_color) #couleur de fond de la fenetre 

        button1=tk.Button(fen_boutons,text="Tabulaire",font=BigBUTTON ,command=lambda:controller.show_frame(TabularPage))
        button1.config(fg=text_button) #couleur du texte
        button1.config(background=bg_button) #couleur de fond du texte
        button1.place(relx=0,rely=1,anchor=tk.SW)

        button2=tk.Button(fen_boutons,text="Image",font=BigBUTTON ,command=lambda:controller.show_frame(TabularPage))
        button2.config(fg=text_button) #couleur du texte
        button2.config(bg=bg_button)
        button2.place(relx=0.5,rely=1,anchor=tk.S)

        button3=tk.Button(fen_boutons,text="Retour",font=BigBUTTON ,command=lambda:controller.show_frame(ExplanationPage))
        button3.config(fg=text_button) #couleur du texte
        button3.config(bg=bg_button)
        button3.place(relx=1,rely=1,anchor=tk.SE)

        fen_Text.pack(pady=100)
        fen_boutons.pack(pady=50)









class TabularPage (tk.Frame): # choix de la méthode d'interpretabilite pour les images

    def __init__(self,parent,controller):

        tk.Frame.__init__(self,parent)
        label=tk.Label(self,text="Choisir la technique d'interprétabilité",font=LARGE_FONT)
        label.config(fg='tomato') #couleur du texte
        label.config(background=frame_color) # couleur de fond du texte
        label.pack(pady=40)

        label=tk.Label(self,text=text_TabularPage,font=SUBTITLE,justify=LEFT)
        label.config(fg=text_color) #couleur du texte
        label.config(background=frame_color) # couleur de fond du texte
        label.pack(pady=10,padx=10)

        # conteneur des boutons
        fen_test = tk.Frame(self, width=1000, height=250,)
        fen_test.configure(bg=frame_color) #couleur de fond de la fenetre

        button0=tk.Button(fen_test,text="LIME",font=BigBUTTON ,command=lambda:controller.show_frame(LIMETab))
        button0.config(fg=text_button) #couleur du texte
        button0.config(bg=Big_button_color)
        button0.place(relx=0,rely=1,anchor=tk.SW)
        
        button1=tk.Button(fen_test,text="Features importance",font=BigBUTTON ,command=lambda:controller.show_frame(FeatureImportance))
        button1.config(fg=text_button) #couleur du texte
        button1.config(bg=Big_button_color)
        button1.place(relx=0.5,rely=1,anchor=tk.S)
        
        button2=tk.Button(fen_test,text="Permutation importance",font=BigBUTTON ,command=lambda:controller.show_frame(PermutationImportance))
        button2.config(fg=text_button) #couleur du texte
        button2.config(bg=Big_button_color)
        button2.place(relx=0.5,rely=0,anchor=tk.N)
        
        button3=tk.Button(fen_test,text="SHAP",font=BigBUTTON ,command=lambda:controller.show_frame(SHAP))
        button3.config(fg=text_button) #couleur du texte
        button3.config(bg=Big_button_color)
        button3.place(relx=0,rely=0,anchor=tk.NW)        
               
        button4=tk.Button(fen_test,text="PDP",font=BigBUTTON ,command=lambda:controller.show_frame(PDP))
        button4.config(fg=text_button) #couleur du texte
        button4.config(bg=Big_button_color)
        button4.place(relx=1,rely=1,anchor=tk.SE)

        # conteneur des boutons
        fen_boutons = tk.Frame(self, width=1000, height=300,)
        fen_boutons.configure(bg=frame_color) #couleur de fond de la fenetre 

        button1=tk.Button(fen_boutons,text="Retour",font=BUTTON ,command=lambda:controller.show_frame(OptionPage))
        button1.config(fg=text_button) #couleur du texte
        button1.config(bg=bg_button)
        button1.place(relx=0,rely=1,anchor=tk.SW)

        button3=tk.Button(fen_boutons,text="Quitter",font=BUTTON ,command=lambda:controller.show_frame(PageQuit))
        button3.config(fg=text_button) #couleur du texte
        button3.config(bg=bg_button)
        button3.place(relx=1,rely=1,anchor=tk.SE)

        fen_test.pack(pady=100)
        fen_boutons.pack(pady=50)










class LIMETab (tk.Frame): # page qui demande confirmation avant de quitter

   def __init__(self,parent,controller):

        tk.Frame.__init__(self,parent)
        label=tk.Label(self,text="LIME",font=LARGE_FONT)
        label.config(fg=text_color) #couleur du texte
        label.config(background=frame_color) # couleur de fond du texte
        label.pack(pady=25)

        # conteneur entry 1 et 2
        fen_test = tk.Frame(self, width=1000, height=100,)
        fen_test.configure(bg=frame_color) #couleur de fond de la fenetre

        label=tk.Label(fen_test,text='Data set',font=BUTTON,justify=LEFT)
        label.config(fg=text_color) #couleur du texte
        label.config(background=frame_color) # couleur de fond du texte
        label.grid(row=0,column=0,padx=5,pady=5)

        label=tk.Label(fen_test,text='Modèle',font=BUTTON,justify=LEFT)
        label.config(fg=text_color) #couleur du texte
        label.config(background=frame_color) # couleur de fond du texte
        label.grid(row=0,column=2,padx=5,pady=5)  

        e_dataset=tk.Entry(fen_test,width=50 )
        e_dataset.grid(row=1,column=0,padx=5,pady=5)

        e_model=tk.Entry(fen_test,width=50 )
        e_model.grid(row=1,column=2,padx=5,pady=5)

        global X_train
        global X_test
        global Y_train
        global Y_test
        global dataset
        global model
        global click
        
        
        X_train, X_test, Y_train, Y_test, dataset, model = 0, 0, 0, 0, 0, 0

        def browsedataset():
            global dataset
            global click
            click = 0

            filename = tk.filedialog.askopenfilename(filetypes=(("Numpy files","*.npy"),("CSV files","*.csv"),("pickle files","*.pkl"),("hdf5 files","*.hdf5"),("hdf5 files","*.h5")))
            e_dataset.insert(tk.END, filename)
            if filename.endswith('.csv'):
                dataset=pd.read_csv(filename)
                 
            elif filename.endswith('.npy'):
                dataset=np.load(filename)

            elif filename.endswith('.pkl'):
                dataset=pd.read_pickle(filename)            

            elif filename.endswith('.h5'):
                dataset=pd.read_hdf(filename) 

            elif filename.endswith('.hdf5'):
                dataset=pd.DataFrame(nparray(h5py.File((filename)))) 

        def browsemodel():

            global model
            global click
            click = 0
            
            filename =tk.filedialog.askopenfilename(filetypes=(("pickle files","*.pkl"), ("sav files", "*.sav")))
            e_model.insert(tk.END, filename)
            model = pickle.load(open(filename, 'rb'))

        Button_entry = tk.Button(fen_test,text="Explorer",font=Browse_BUTTON ,command=browsedataset)
        Button_entry.config(fg=Browse_BUTTON_ft_color) #couleur du textelambda:
        Button_entry.config(bg=Browse_BUTTON_bg_color)
        Button_entry.grid(row=1,column=1,padx=5,pady=5)

        Button_emod = tk.Button(fen_test,text="Explorer",font=Browse_BUTTON ,command=browsemodel)
        Button_emod.config(fg=Browse_BUTTON_ft_color) #couleur du textelambda:
        Button_emod.config(bg=Browse_BUTTON_bg_color)
        Button_emod.grid(row=1,column=3,padx=5,pady=5)

        # conteneur entry 3, 4, 5, 6
        fen_test2 = tk.Frame(self, width=1000, height=100,)
        fen_test2.configure(bg=frame_color) #couleur de fond de la fenetre

        label=tk.Label(fen_test2,text='X train',font=BUTTON,justify=LEFT)
        label.config(fg=text_color) #couleur du texte
        label.config(background=frame_color) # couleur de fond du texte
        label.grid(row=0,column=0,padx=5,pady=5)         

        e_dataXtrain=tk.Entry(fen_test2,width=10 )
        e_dataXtrain.grid(row=0,column=1,padx=5,pady=5)  

        label=tk.Label(fen_test2,text='Y train',font=BUTTON,justify=LEFT)
        label.config(fg=text_color) #couleur du texte
        label.config(background=frame_color) # couleur de fond du texte
        label.grid(row=0,column=3,padx=5,pady=5)         

        e_dataYtrain=tk.Entry(fen_test2,width=10 )
        e_dataYtrain.grid(row=0,column=4,padx=5,pady=5)      

        label=tk.Label(fen_test2,text='X test',font=BUTTON,justify=LEFT)
        label.config(fg=text_color) #couleur du texte
        label.config(background=frame_color) # couleur de fond du texte
        label.grid(row=0,column=6,padx=5,pady=5)         

        e_dataXtest=tk.Entry(fen_test2,width=10 )
        e_dataXtest.grid(row=0,column=7,padx=5,pady=5)

        label=tk.Label(fen_test2,text='Y test',font=BUTTON,justify=LEFT)
        label.config(fg=text_color) #couleur du texte
        label.config(background=frame_color) # couleur de fond du texte
        label.grid(row=0,column=9,padx=5,pady=5)         

        e_dataYtest=tk.Entry(fen_test2,width=10 )
        e_dataYtest.grid(row=0,column=10,padx=5,pady=5)

        def browseX_train():
            
            global X_train
            global click
            click = 0
            
            filetrain = tk.filedialog.askopenfilename(filetypes=(("Numpy files","*.npy"),("CSV files","*.csv"),("pickle files","*.pkl"),("hdf5 files","*.hdf5"),("hdf5 files","*.h5")))
            e_dataXtrain.insert(tk.END, filetrain) # add this

            if filetrain.endswith('.csv'):
                X_train=pd.read_csv(filetrain)                    

            elif filetrain.endswith('.npy'):
                X_train=np.load(filetrain)
                
            elif filetrain.endswith('.pkl'):
                X_train=pd.read_pickle(filetrain) 
                               
            elif filetrain.endswith('.h5'):
                X_train=pd.read_hdf(filetrain) 
                               
            elif filetrain.endswith('.hdf5'):
                X_train=pd.DataFrame(nparray(h5py.File((filetrain))))

            if isinstance(X_train, np.ndarray):
                X_train = pd.DataFrame(X_train)
            print(X_train)

        def browseY_train():
            
            global Y_train
            global click
            click = 0            
            
            filetrain = tk.filedialog.askopenfilename(filetypes=(("Numpy files","*.npy"),("CSV files","*.csv"),("pickle files","*.pkl"),("hdf5 files","*.hdf5"),("hdf5 files","*.h5")))
            e_dataYtrain.insert(tk.END, filetrain) # add this

            if filetrain.endswith('.csv'):
                Y_train=pd.read_csv(filetrain)            
                                       
            elif filetrain.endswith('.npy'):
                Y_train=np.load(filetrain)
                
            elif filetrain.endswith('.pkl'):
                Y_train=pd.read_pickle(filetrain) 
                               
            elif filetrain.endswith('.h5'):
                Y_train=pd.read_hdf(filetrain) 
                               
            elif filetrain.endswith('.hdf5'):
                Y_train=pd.DataFrame(nparray(h5py.File((filetrain)))) 
            
            if isinstance(Y_train, np.ndarray):
                Y_train = pd.DataFrame(Y_train)

        def browseX_test():
           
            global X_test
            global click
            click = 0

            filetest = tk.filedialog.askopenfilename(filetypes=(("Numpy files","*.npy"),("CSV files","*.csv"),("pickle files","*.pkl"),("hdf5 files","*.hdf5"),("hdf5 files","*.h5")))
            print(filetest.endswith)
            
            e_dataXtest.insert(tk.END, filetest) # add this
           
            if filetest.endswith('.csv'):
                X_test=pd.read_csv(filetest)
                                       
            elif filetest.endswith('.npy'):
                X_test=np.load(filetest)
                
            elif filetest.endswith('.pkl'):
                X_test=pd.read_pickle(filetest) 
                               
            elif filetest.endswith('.h5'):
                X_test=pd.read_hdf(filetest) 
                               
            elif filetest.endswith('.hdf5'):
                X_test=pd.DataFrame(nparray(h5py.File((filetest))))  
            
            if isinstance(X_test, np.ndarray):
                X_test = pd.DataFrame(X_test)

        def browseY_test():

            global Y_test
            global click
            click = 0
            
            filetest = tk.filedialog.askopenfilename(filetypes=(("Numpy files","*.npy"),("CSV files","*.csv"),("pickle files","*.pkl"),("hdf5 files","*.hdf5"),("hdf5 files","*.h5")))
            e_dataYtest.insert(tk.END, filetest) # add this
            if filetest.endswith('.csv'):
                Y_test=pd.read_csv(filetest)

            elif filetest.endswith('.npy'):
                Y_test=np.load(filetest)
                
            elif filetest.endswith('.pkl'):
                Y_test=pd.read_pickle(filetest) 
                               
            elif filetest.endswith('.h5'):
                Y_test=pd.read_hdf(filetest) 
                               
            elif filetest.endswith('.hdf5'):
                Y_test=pd.DataFrame(nparray(h5py.File((filetest)))) 
        
            if isinstance(Y_test, np.ndarray):
                Y_test = pd.DataFrame(Y_test)


        Button_etrain = tk.Button(fen_test2,text="Explorer",font=Browse_BUTTON ,command=browseX_train)
        Button_etrain.config(fg=Browse_BUTTON_ft_color) #couleur du texte lambda:
        Button_etrain.config(bg=Browse_BUTTON_bg_color)
        Button_etrain.grid(row=0,column=2,padx=5,pady=5)

        Button_etrain = tk.Button(fen_test2,text="Explorer",font=Browse_BUTTON ,command=browseY_train)
        Button_etrain.config(fg=Browse_BUTTON_ft_color) #couleur du texte lambda:
        Button_etrain.config(bg=Browse_BUTTON_bg_color)
        Button_etrain.grid(row=0,column=5,padx=5,pady=5)
        
        Button_etrain = tk.Button(fen_test2,text="Explorer",font=Browse_BUTTON ,command=browseX_test)
        Button_etrain.config(fg=Browse_BUTTON_ft_color) #couleur du texte lambda:
        Button_etrain.config(bg=Browse_BUTTON_bg_color)
        Button_etrain.grid(row=0,column=8,padx=5,pady=5)

        Button_etrain = tk.Button(fen_test2,text="Explorer",font=Browse_BUTTON ,command=browseY_test)
        Button_etrain.config(fg=Browse_BUTTON_ft_color) #couleur du texte lambda:
        Button_etrain.config(bg=Browse_BUTTON_bg_color)
        Button_etrain.grid(row=0,column=11,padx=5,pady=5)

        global currentGraph
        global currentdirectory

        #conteneur canvas   
        fen_can = tk.Frame(self, width=1000, height=100,)
        fen_can.configure(bg=frame_color) #couleur de fond de la fenetre
        
        canevas_lime = tk.Canvas(fen_can, width = 350, height = 350, bg = 'white')
        canevas_lime.grid(row = 0, column = 0)
        
        
        def check():
            global dataset
            if ( (type(dataset)!=int) and (type(model)!=int) and (type(X_train) != int) and (type(X_test)!=int) and (type(Y_train) != int) and (type(Y_test) != int) ) :
                LIME()

        def LIME () :

            global currentdirectory
            global X_train
            global X_test
            global model

            explainer = lime.lime_tabular.LimeTabularExplainer(X_train.values, mode='regression',training_labels=Y_train,feature_names=dataset.columns)
            explanation = explainer.explain_instance(X_test.values[0], model.predict)
            explanation.as_pyplot_figure().set_size_inches((4, 4))
            plt.tight_layout()
            plt.savefig(currentdirectory + "/lime.png")
            plt.clf()
            plotLIME()


        def plotLIME(): 

		
            global currentGraph 
            global currentdirectory
            global click
            if (click==0) :                       
                canvas_lime = tk.Canvas(fen_can,width=250, height=250, bg="white") 
                canvas_lime.grid(row=0, column=0)

                # Clear all graphs drawn in figure 
                photo = tk.PhotoImage(file=currentdirectory + "/lime.png")
                labelimg = tk.Label(canvas_lime, image=photo)
                labelimg.image = photo
                labelimg.pack(pady=0, padx=0)
                click+=1

            X_train, X_test, Y_train, Y_test, dataset, model = 0, 0, 0, 0, 0, 0 

        # Create a tkinter button at the bottom of the window and link it with the updateGraph function 
        tk.Button(fen_can,text="Générer",command=check).grid(row=0, column=1) 

        # conteneur des boutons
        fen_boutons = tk.Frame(self, width=500, height=200,)
        fen_boutons.configure(bg=frame_color) #couleur de fond de la fenetre 

        button1=tk.Button(fen_boutons,text="Retour",font=BUTTON ,command=lambda:controller.show_frame(TabularPage))
        button1.config(fg=text_button) #couleur du texte
        button1.config(bg=bg_button)
        button1.place(relx=0,rely=1,anchor=tk.SW)

        button3=tk.Button(fen_boutons,text="Quitter",font=BUTTON ,command=lambda:controller.show_frame(PageQuit))
        button3.config(fg=text_button) #couleur du texte
        button3.config(bg=bg_button)
        button3.place(relx=1,rely=1,anchor=tk.SE)

        fen_test.pack(pady=2)
        fen_test2.pack(pady=2)
        fen_can.pack(pady=2)
        fen_boutons.pack(pady=30)
 
 
 
 
 
 
class PDP (tk.Frame): # page qui demande confirmation avant de quitter

   def __init__(self,parent,controller):

        tk.Frame.__init__(self,parent)
        label=tk.Label(self,text="PDP",font=LARGE_FONT)
        label.config(fg=text_color) #couleur du texte
        label.config(background=frame_color) # couleur de fond du texte
        label.pack(pady=25)

        # conteneur entry 1 et 2
        fen_test = tk.Frame(self, width=1000, height=100,)
        fen_test.configure(bg=frame_color) #couleur de fond de la fenetre

        label=tk.Label(fen_test,text='Data set',font=BUTTON,justify=LEFT)
        label.config(fg=text_color) #couleur du texte
        label.config(background=frame_color) # couleur de fond du texte
        label.grid(row=0,column=0,padx=5,pady=5)

        label=tk.Label(fen_test,text='Modèle',font=BUTTON,justify=LEFT)
        label.config(fg=text_color) #couleur du texte
        label.config(background=frame_color) # couleur de fond du texte
        label.grid(row=0,column=2,padx=5,pady=5)  

        e_dataset=tk.Entry(fen_test,width=50 )
        e_dataset.grid(row=1,column=0,padx=5,pady=5)

        e_model=tk.Entry(fen_test,width=50 )
        e_model.grid(row=1,column=2,padx=5,pady=5)

        global X_train
        global X_test
        global Y_train
        global Y_test
        global dataset
        global model
        global click
        
        click=0
        X_train, X_test, Y_train, Y_test, dataset, model = 0, 0, 0, 0, 0, 0

        def browsedataset():
            global dataset
            global click
            click = 0

            filename = tk.filedialog.askopenfilename(filetypes=(("Numpy files","*.npy"),("CSV files","*.csv"),("pickle files","*.pkl"),("hdf5 files","*.hdf5"),("hdf5 files","*.h5")))
            e_dataset.insert(tk.END, filename)
            if filename.endswith('.csv'):
                dataset=pd.read_csv(filename)
                 
            elif filename.endswith('.npy'):
                dataset=np.load(filename)

            elif filename.endswith('.pkl'):
                dataset=pd.read_pickle(filename)            

            elif filename.endswith('.h5'):
                dataset=pd.read_hdf(filename) 

            elif filename.endswith('.hdf5'):
                dataset=pd.DataFrame(nparray(h5py.File((filename)))) 

        def browsemodel():

            global model
            global click
            click = 0
                    
            filename =tk.filedialog.askopenfilename(filetypes=(("pickle files","*.pkl"), ("sav files", "*.sav")))
            e_model.insert(tk.END, filename)
            model = pickle.load(open(filename, 'rb'))

        Button_entry = tk.Button(fen_test,text="Explorer",font=Browse_BUTTON ,command=browsedataset)
        Button_entry.config(fg=Browse_BUTTON_ft_color) #couleur du textelambda:
        Button_entry.config(bg=Browse_BUTTON_bg_color)
        Button_entry.grid(row=1,column=1,padx=5,pady=5)

        Button_emod = tk.Button(fen_test,text="Explorer",font=Browse_BUTTON ,command=browsemodel)
        Button_emod.config(fg=Browse_BUTTON_ft_color) #couleur du textelambda:
        Button_emod.config(bg=Browse_BUTTON_bg_color)
        Button_emod.grid(row=1,column=3,padx=5,pady=5)

        # conteneur entry 3, 4, 5, 6
        fen_test2 = tk.Frame(self, width=1000, height=100,)
        fen_test2.configure(bg=frame_color) #couleur de fond de la fenetre

        label=tk.Label(fen_test2,text='X train',font=BUTTON,justify=LEFT)
        label.config(fg=text_color) #couleur du texte
        label.config(background=frame_color) # couleur de fond du texte
        label.grid(row=0,column=0,padx=5,pady=5)         

        e_dataXtrain=tk.Entry(fen_test2,width=10 )
        e_dataXtrain.grid(row=0,column=1,padx=5,pady=5)  

        label=tk.Label(fen_test2,text='Y train',font=BUTTON,justify=LEFT)
        label.config(fg=text_color) #couleur du texte
        label.config(background=frame_color) # couleur de fond du texte
        label.grid(row=0,column=3,padx=5,pady=5)         

        e_dataYtrain=tk.Entry(fen_test2,width=10 )
        e_dataYtrain.grid(row=0,column=4,padx=5,pady=5)      

        label=tk.Label(fen_test2,text='X test',font=BUTTON,justify=LEFT)
        label.config(fg=text_color) #couleur du texte
        label.config(background=frame_color) # couleur de fond du texte
        label.grid(row=0,column=6,padx=5,pady=5)         

        e_dataXtest=tk.Entry(fen_test2,width=10 )
        e_dataXtest.grid(row=0,column=7,padx=5,pady=5)

        label=tk.Label(fen_test2,text='Y test',font=BUTTON,justify=LEFT)
        label.config(fg=text_color) #couleur du texte
        label.config(background=frame_color) # couleur de fond du texte
        label.grid(row=0,column=9,padx=5,pady=5)         

        e_dataYtest=tk.Entry(fen_test2,width=10 )
        e_dataYtest.grid(row=0,column=10,padx=5,pady=5)

        def browseX_train():
            
            global X_train
            global click
            click = 0
            
            filetrain = tk.filedialog.askopenfilename(filetypes=(("Numpy files","*.npy"),("CSV files","*.csv"),("pickle files","*.pkl"),("hdf5 files","*.hdf5"),("hdf5 files","*.h5")))
            e_dataXtrain.insert(tk.END, filetrain) # add this

            if filetrain.endswith('.csv'):
                X_train=pd.read_csv(filetrain)                    

            elif filetrain.endswith('.npy'):
                X_train=np.load(filetrain)
                
            elif filetrain.endswith('.pkl'):
                X_train=pd.read_pickle(filetrain) 
                               
            elif filetrain.endswith('.h5'):
                X_train=pd.read_hdf(filetrain) 
                               
            elif filetrain.endswith('.hdf5'):
                X_train=pd.DataFrame(nparray(h5py.File((filetrain))))

            if isinstance(X_train, np.ndarray):
                X_train = pd.DataFrame(X_train)

        def browseY_train():
            
            global Y_train
            global click
            click = 0
            
            filetrain = tk.filedialog.askopenfilename(filetypes=(("Numpy files","*.npy"),("CSV files","*.csv"),("pickle files","*.pkl"),("hdf5 files","*.hdf5"),("hdf5 files","*.h5")))
            e_dataYtrain.insert(tk.END, filetrain) # add this

            if filetrain.endswith('.csv'):
                Y_train=pd.read_csv(filetrain)            
                                       
            elif filetrain.endswith('.npy'):
                Y_train=np.load(filetrain)
                
            elif filetrain.endswith('.pkl'):
                Y_train=pd.read_pickle(filetrain) 
                               
            elif filetrain.endswith('.h5'):
                Y_train=pd.read_hdf(filetrain) 
                               
            elif filetrain.endswith('.hdf5'):
                Y_train=pd.DataFrame(nparray(h5py.File((filetrain)))) 
            
            if isinstance(Y_train, np.ndarray):
                Y_train = pd.DataFrame(Y_train)

        def browseX_test():
           
            global X_test
            global click
            click = 0
            
            filetest = tk.filedialog.askopenfilename(filetypes=(("Numpy files","*.npy"),("CSV files","*.csv"),("pickle files","*.pkl"),("hdf5 files","*.hdf5"),("hdf5 files","*.h5")))
            print(filetest.endswith)
            
            e_dataXtest.insert(tk.END, filetest) # add this
           
            if filetest.endswith('.csv'):
                X_test=pd.read_csv(filetest)
                                       
            elif filetest.endswith('.npy'):
                X_test=np.load(filetest)
                
            elif filetest.endswith('.pkl'):
                X_test=pd.read_pickle(filetest) 
                               
            elif filetest.endswith('.h5'):
                X_test=pd.read_hdf(filetest) 
                               
            elif filetest.endswith('.hdf5'):
                X_test=pd.DataFrame(nparray(h5py.File((filetest))))  
            
            if isinstance(X_test, np.ndarray):
                X_test = pd.DataFrame(X_test)

        def browseY_test():

            global Y_test
            global click
            click = 0
            
            filetest = tk.filedialog.askopenfilename(filetypes=(("Numpy files","*.npy"),("CSV files","*.csv"),("pickle files","*.pkl"),("hdf5 files","*.hdf5"),("hdf5 files","*.h5")))
            e_dataYtest.insert(tk.END, filetest) # add this
            if filetest.endswith('.csv'):
                Y_test=pd.read_csv(filetest)

            elif filetest.endswith('.npy'):
                Y_test=np.load(filetest)
                
            elif filetest.endswith('.pkl'):
                Y_test=pd.read_pickle(filetest) 
                               
            elif filetest.endswith('.h5'):
                Y_test=pd.read_hdf(filetest) 
                               
            elif filetest.endswith('.hdf5'):
                Y_test=pd.DataFrame(nparray(h5py.File((filetest)))) 
        
            if isinstance(Y_test, np.ndarray):
                Y_test = pd.DataFrame(Y_test)


        Button_etrain = tk.Button(fen_test2,text="Explorer",font=Browse_BUTTON ,command=browseX_train)
        Button_etrain.config(fg=Browse_BUTTON_ft_color) #couleur du texte lambda:
        Button_etrain.config(bg=Browse_BUTTON_bg_color)
        Button_etrain.grid(row=0,column=2,padx=5,pady=5)

        Button_etrain = tk.Button(fen_test2,text="Explorer",font=Browse_BUTTON ,command=browseY_train)
        Button_etrain.config(fg=Browse_BUTTON_ft_color) #couleur du texte lambda:
        Button_etrain.config(bg=Browse_BUTTON_bg_color)
        Button_etrain.grid(row=0,column=5,padx=5,pady=5)
        
        Button_etrain = tk.Button(fen_test2,text="Explorer",font=Browse_BUTTON ,command=browseX_test)
        Button_etrain.config(fg=Browse_BUTTON_ft_color) #couleur du texte lambda:
        Button_etrain.config(bg=Browse_BUTTON_bg_color)
        Button_etrain.grid(row=0,column=8,padx=5,pady=5)

        Button_etrain = tk.Button(fen_test2,text="Explorer",font=Browse_BUTTON ,command=browseY_test)
        Button_etrain.config(fg=Browse_BUTTON_ft_color) #couleur du texte lambda:
        Button_etrain.config(bg=Browse_BUTTON_bg_color)
        Button_etrain.grid(row=0,column=11,padx=5,pady=5)

        global currentGraph
        global currentdirectory

        #conteneur canvas   
        fen_can = tk.Frame(self, width=1000, height=100,)
        fen_can.configure(bg=frame_color) #couleur de fond de la fenetre

        canvas_PDP = tk.Canvas(fen_can,width=350, height=350, bg="white") 
        canvas_PDP.grid(row=0, column=0)
        
        def check():
            global dataset
            if ( (type(dataset)!=int) and (type(model)!=int) and (type(X_train) != int) and (type(X_test)!=int) and (type(Y_train) != int) and (type(Y_test) != int) ) :
                PDP()

        def PDP() :

            global currentdirectory
            global X_train
            global X_test
            global model

            numero = 0

            for column in X_train.columns :

                pdp_goals = pdp.pdp_isolate(model=model, dataset = X_train, model_features=X_train.columns.tolist(), feature=column)

                # plot it
                fig = pdp.pdp_plot(pdp_goals, column, plot_params = {'title': 'PDP for feature "%s"' % column, 'subtitle':''})
                fig[0].set_size_inches(4, 4)
                plt.tight_layout()
                plt.savefig(currentdirectory + "/pdp{}.png".format(numero))
                numero += 1
                plt.clf()
            plotPDP()
            
        def plotPDP(): 

            global currentGraph 
            global currentdirectory
            global click
            if (click==0) :
                # Clear all graphs drawn in figure 
                photo = tk.PhotoImage(file=currentdirectory + "/pdp{}.png".format(0))
                labelimg = tk.Label(canvas_PDP, image=photo)
                labelimg.image = photo
                labelimg.pack(pady=0, padx=0)
                click+=1

        # Create a tkinter button at the bottom of the window and link it with the updateGraph function 
        tk.Button(fen_can,text="Générer",command=check).grid(row=0, column=1) 

        # conteneur des boutons
        fen_boutons = tk.Frame(self, width=500, height=200,)
        fen_boutons.configure(bg=frame_color) #couleur de fond de la fenetre 

        button1=tk.Button(fen_boutons,text="Retour",font=BUTTON ,command=lambda:controller.show_frame(TabularPage))
        button1.config(fg=text_button) #couleur du texte
        button1.config(bg=bg_button)
        button1.place(relx=0,rely=1,anchor=tk.SW)

        button3=tk.Button(fen_boutons,text="Quitter",font=BUTTON ,command=lambda:controller.show_frame(PageQuit))
        button3.config(fg=text_button) #couleur du texte
        button3.config(bg=bg_button)
        button3.place(relx=1,rely=1,anchor=tk.SE)

        fen_test.pack(pady=2)
        fen_test2.pack(pady=2)
        fen_can.pack(pady=2)
        fen_boutons.pack(pady=30)









  
class FeatureImportance (tk.Frame): # page qui demande confirmation avant de quitter

   def __init__(self,parent,controller):

        tk.Frame.__init__(self,parent)
        label=tk.Label(self,text="Feature Importance",font=LARGE_FONT)
        label.config(fg='tomato') #couleur du texte
        label.config(background=frame_color) # couleur de fond du texte
        label.pack(pady=25)

        # conteneur entry 1 et 2
        fen_test = tk.Frame(self, width=1000, height=100,)
        fen_test.configure(bg=frame_color) #couleur de fond de la fenetre

        label=tk.Label(fen_test,text='Data set',font=BUTTON,justify=LEFT)
        label.config(fg=text_color) #couleur du texte
        label.config(background=frame_color) # couleur de fond du texte
        label.grid(row=0,column=0,padx=5,pady=5)

        label=tk.Label(fen_test,text='Modèle',font=BUTTON,justify=LEFT)
        label.config(fg=text_color) #couleur du texte
        label.config(background=frame_color) # couleur de fond du texte
        label.grid(row=0,column=2,padx=5,pady=5)  

        e_dataset=tk.Entry(fen_test,width=15 )
        e_dataset.grid(row=1,column=0,padx=5,pady=5)

        e_model=tk.Entry(fen_test,width=15 )
        e_model.grid(row=1,column=2,padx=5,pady=5)

        global X_train
        global X_test
        global Y_train
        global Y_test
        global dataset
        global model
        global click
        
        click=0
        X_train, X_test, Y_train, Y_test, dataset, model = 0, 0, 0, 0, 0, 0

        def browsedataset():
            global dataset
            global click
            click = 0

            filename = tk.filedialog.askopenfilename(filetypes=(("Numpy files","*.npy"),("CSV files","*.csv"),("pickle files","*.pkl"),("hdf5 files","*.hdf5"),("hdf5 files","*.h5")))
            e_dataset.insert(tk.END, filename)
            if filename.endswith('.csv'):
                dataset=pd.read_csv(filename)
                 
            elif filename.endswith('.npy'):
                dataset=np.load(filename)

            elif filename.endswith('.pkl'):
                dataset=pd.read_pickle(filename)            

            elif filename.endswith('.h5'):
                dataset=pd.read_hdf(filename) 

            elif filename.endswith('.hdf5'):
                dataset=pd.DataFrame(nparray(h5py.File((filename)))) 

        def browsemodel():

            global model
            global click
            click = 0
                    
            filename =tk.filedialog.askopenfilename(filetypes=(("pickle files","*.pkl"), ("sav files", "*.sav")))
            e_model.insert(tk.END, filename)
            model = pickle.load(open(filename, 'rb'))

        Button_entry = tk.Button(fen_test,text="Explorer",font=Browse_BUTTON ,command=browsedataset)
        Button_entry.config(fg=Browse_BUTTON_ft_color) #couleur du textelambda:
        Button_entry.config(bg=Browse_BUTTON_bg_color)
        Button_entry.grid(row=1,column=1,padx=5,pady=5)

        Button_emod = tk.Button(fen_test,text="Explorer",font=Browse_BUTTON ,command=browsemodel)
        Button_emod.config(fg=Browse_BUTTON_ft_color) #couleur du textelambda:
        Button_emod.config(bg=Browse_BUTTON_bg_color)
        Button_emod.grid(row=1,column=3,padx=5,pady=5)

        # conteneur entry 3, 4, 5, 6
        fen_test2 = tk.Frame(self, width=1000, height=100,)
        fen_test2.configure(bg=frame_color) #couleur de fond de la fenetre

        label=tk.Label(fen_test2,text='X train',font=BUTTON,justify=LEFT)
        label.config(fg=text_color) #couleur du texte
        label.config(background=frame_color) # couleur de fond du texte
        label.grid(row=0,column=0,padx=5,pady=5)         

        e_dataXtrain=tk.Entry(fen_test2,width=15 )
        e_dataXtrain.grid(row=0,column=1,padx=5,pady=5)  

        label=tk.Label(fen_test2,text='Y train',font=BUTTON,justify=LEFT)
        label.config(fg=text_color) #couleur du texte
        label.config(background=frame_color) # couleur de fond du texte
        label.grid(row=0,column=3,padx=5,pady=5)         

        e_dataYtrain=tk.Entry(fen_test2,width=15 )
        e_dataYtrain.grid(row=0,column=4,padx=5,pady=5)      

        label=tk.Label(fen_test2,text='X test',font=BUTTON,justify=LEFT)
        label.config(fg=text_color) #couleur du texte
        label.config(background=frame_color) # couleur de fond du texte
        label.grid(row=0,column=6,padx=5,pady=5)         

        e_dataXtest=tk.Entry(fen_test2,width=15 )
        e_dataXtest.grid(row=0,column=7,padx=5,pady=5)

        label=tk.Label(fen_test2,text='Y test',font=BUTTON,justify=LEFT)
        label.config(fg=text_color) #couleur du texte
        label.config(background=frame_color) # couleur de fond du texte
        label.grid(row=0,column=9,padx=5,pady=5)         

        e_dataYtest=tk.Entry(fen_test2,width=15 )
        e_dataYtest.grid(row=0,column=10,padx=5,pady=5)

        def browseX_train():
            
            global X_train
            global click
            click = 0
            
            filetrain = tk.filedialog.askopenfilename(filetypes=(("Numpy files","*.npy"),("CSV files","*.csv"),("pickle files","*.pkl"),("hdf5 files","*.hdf5"),("hdf5 files","*.h5")))
            e_dataXtrain.insert(tk.END, filetrain) # add this

            if filetrain.endswith('.csv'):
                X_train=pd.read_csv(filetrain)                    

            elif filetrain.endswith('.npy'):
                X_train=np.load(filetrain)
                
            elif filetrain.endswith('.pkl'):
                X_train=pd.read_pickle(filetrain) 
                               
            elif filetrain.endswith('.h5'):
                X_train=pd.read_hdf(filetrain) 
                               
            elif filetrain.endswith('.hdf5'):
                X_train=pd.DataFrame(nparray(h5py.File((filetrain))))

            if isinstance(X_train, np.ndarray):
                X_train = pd.DataFrame(X_train)

        def browseY_train():
            
            global Y_train
            global click
            click = 0
            
            filetrain = tk.filedialog.askopenfilename(filetypes=(("Numpy files","*.npy"),("CSV files","*.csv"),("pickle files","*.pkl"),("hdf5 files","*.hdf5"),("hdf5 files","*.h5")))
            e_dataYtrain.insert(tk.END, filetrain) # add this

            if filetrain.endswith('.csv'):
                Y_train=pd.read_csv(filetrain)            
                                       
            elif filetrain.endswith('.npy'):
                Y_train=np.load(filetrain)
                
            elif filetrain.endswith('.pkl'):
                Y_train=pd.read_pickle(filetrain) 
                               
            elif filetrain.endswith('.h5'):
                Y_train=pd.read_hdf(filetrain) 
                               
            elif filetrain.endswith('.hdf5'):
                Y_train=pd.DataFrame(nparray(h5py.File((filetrain)))) 
            
            if isinstance(Y_train, np.ndarray):
                Y_train = pd.DataFrame(Y_train)

        def browseX_test():
           
            global X_test
            global click
            click = 0
            
            filetest = tk.filedialog.askopenfilename(filetypes=(("Numpy files","*.npy"),("CSV files","*.csv"),("pickle files","*.pkl"),("hdf5 files","*.hdf5"),("hdf5 files","*.h5")))
            print(filetest.endswith)
            
            e_dataXtest.insert(tk.END, filetest) # add this
           
            if filetest.endswith('.csv'):
                X_test=pd.read_csv(filetest)
                                       
            elif filetest.endswith('.npy'):
                X_test=np.load(filetest)
                
            elif filetest.endswith('.pkl'):
                X_test=pd.read_pickle(filetest) 
                               
            elif filetest.endswith('.h5'):
                X_test=pd.read_hdf(filetest) 
                               
            elif filetest.endswith('.hdf5'):
                X_test=pd.DataFrame(nparray(h5py.File((filetest))))  
            
            if isinstance(X_test, np.ndarray):
                X_test = pd.DataFrame(X_test)

        def browseY_test():

            global Y_test
            global click
            click = 0
            
            filetest = tk.filedialog.askopenfilename(filetypes=(("Numpy files","*.npy"),("CSV files","*.csv"),("pickle files","*.pkl"),("hdf5 files","*.hdf5"),("hdf5 files","*.h5")))
            e_dataYtest.insert(tk.END, filetest) # add this
            if filetest.endswith('.csv'):
                Y_test=pd.read_csv(filetest)

            elif filetest.endswith('.npy'):
                Y_test=np.load(filetest)
                
            elif filetest.endswith('.pkl'):
                Y_test=pd.read_pickle(filetest) 
                               
            elif filetest.endswith('.h5'):
                Y_test=pd.read_hdf(filetest) 
                               
            elif filetest.endswith('.hdf5'):
                Y_test=pd.DataFrame(nparray(h5py.File((filetest)))) 
        
            if isinstance(Y_test, np.ndarray):
                Y_test = pd.DataFrame(Y_test)


        Button_etrain = tk.Button(fen_test2,text="Explorer",font=Browse_BUTTON ,command=browseX_train)
        Button_etrain.config(fg=Browse_BUTTON_ft_color) #couleur du texte lambda:
        Button_etrain.config(bg=Browse_BUTTON_bg_color)
        Button_etrain.grid(row=0,column=2,padx=5,pady=5)

        Button_etrain = tk.Button(fen_test2,text="Explorer",font=Browse_BUTTON ,command=browseY_train)
        Button_etrain.config(fg=Browse_BUTTON_ft_color) #couleur du texte lambda:
        Button_etrain.config(bg=Browse_BUTTON_bg_color)
        Button_etrain.grid(row=0,column=5,padx=5,pady=5)
        
        Button_etrain = tk.Button(fen_test2,text="Explorer",font=Browse_BUTTON ,command=browseX_test)
        Button_etrain.config(fg=Browse_BUTTON_ft_color) #couleur du texte lambda:
        Button_etrain.config(bg=Browse_BUTTON_bg_color)
        Button_etrain.grid(row=0,column=8,padx=5,pady=5)

        Button_etrain = tk.Button(fen_test2,text="Explorer",font=Browse_BUTTON ,command=browseY_test)
        Button_etrain.config(fg=Browse_BUTTON_ft_color) #couleur du texte lambda:
        Button_etrain.config(bg=Browse_BUTTON_bg_color)
        Button_etrain.grid(row=0,column=11,padx=5,pady=5)

        global currentGraph
        global currentdirectory

        #conteneur canvas   
        fen_can = tk.Frame(self, width=1000, height=100,)
        fen_can.configure(bg=frame_color) #couleur de fond de la fenetre

        canvas_FeatureImportance = tk.Canvas(fen_can,width=350, height=350, bg="white") 
        canvas_FeatureImportance.grid(row=0, column=0)
        
        def check():
            global data_frame
            if ( (type(dataset)!=int) and (type(model)!=int) and (type(X_train) != int) and (type(X_test)!=int) and (type(Y_train) != int) and (type(Y_test) != int) ) :
                Feature_importance()

        def Feature_importance() :
            global currentdirectory
            global X_train
            global X_test
            global model
            importance = model.feature_importances_
           
            # plot feature importance
            fig= plt.figure(figsize = (4,4))
            ax = fig.add_subplot(111)
            
            plt.bar([x for x in range(len(importance))], importance)
            plt.xlabel("Features")
            plt.tight_layout()
            plt.savefig(currentdirectory + "/feature_importance.png")
            plt.clf()
            plotFeatureImportance()

        def plotFeatureImportance(): 
		
            global currentGraph 
            global currentdirectory
            global click
            if (click==0) :
                # Clear all graphs drawn in figure 
                photo = tk.PhotoImage(file=currentdirectory + "/feature_importance.png")
                labelimg = tk.Label(canvas_FeatureImportance, image=photo)
                labelimg.image = photo
                labelimg.pack(pady=0, padx=0)
                click+=1


        # Create a tkinter button at the bottom of the window and link it with the updateGraph function 
        tk.Button(fen_can,text="Générer",command=check).grid(row=0, column=1) 

        # conteneur des boutons
        fen_boutons = tk.Frame(self, width=500, height=200,)
        fen_boutons.configure(bg=frame_color) #couleur de fond de la fenetre 

        button1=tk.Button(fen_boutons,text="Retour",font=BUTTON ,command=lambda:controller.show_frame(TabularPage))
        button1.config(fg=text_button) #couleur du texte
        button1.config(bg=bg_button)
        button1.place(relx=0,rely=1,anchor=tk.SW)

        button3=tk.Button(fen_boutons,text="Quitter",font=BUTTON ,command=lambda:controller.show_frame(PageQuit))
        button3.config(fg=text_button) #couleur du texte
        button3.config(bg=bg_button)
        button3.place(relx=1,rely=1,anchor=tk.SE)

        fen_test.pack(pady=2)
        fen_test2.pack(pady=2)
        fen_can.pack(pady=2)
        fen_boutons.pack(pady=30)








class PermutationImportance (tk.Frame): # page qui demande confirmation avant de quitter

   def __init__(self,parent,controller):

        tk.Frame.__init__(self,parent)
        label=tk.Label(self,text="Permutation Importance",font=LARGE_FONT)
        label.config(fg=text_color) #couleur du texte
        label.config(background=frame_color) # couleur de fond du texte
        label.pack(pady=25)

        # conteneur entry 1 et 2
        fen_test = tk.Frame(self, width=1000, height=100,)
        fen_test.configure(bg=frame_color) #couleur de fond de la fenetre

        label=tk.Label(fen_test,text='Data set',font=BUTTON,justify=LEFT)
        label.config(fg=text_color) #couleur du texte
        label.config(background=frame_color) # couleur de fond du texte
        label.grid(row=0,column=0,padx=5,pady=5)

        label=tk.Label(fen_test,text='Modèle',font=BUTTON,justify=LEFT)
        label.config(fg=text_color) #couleur du texte
        label.config(background=frame_color) # couleur de fond du texte
        label.grid(row=0,column=2,padx=5,pady=5)  

        e_dataset=tk.Entry(fen_test,width=50 )
        e_dataset.grid(row=1,column=0,padx=5,pady=5)

        e_model=tk.Entry(fen_test,width=50 )
        e_model.grid(row=1,column=2,padx=5,pady=5)

        global X_train
        global X_test
        global Y_train
        global Y_test
        global dataset
        global model
        global click
        
        click=0
        X_train, X_test, Y_train, Y_test, dataset, model = 0, 0, 0, 0, 0, 0

        def browsedataset():
            global dataset
            global click
            click = 0

            filename = tk.filedialog.askopenfilename(filetypes=(("Numpy files","*.npy"),("CSV files","*.csv"),("pickle files","*.pkl"),("hdf5 files","*.hdf5"),("hdf5 files","*.h5")))
            e_dataset.insert(tk.END, filename)
            if filename.endswith('.csv'):
                dataset=pd.read_csv(filename)
                 
            elif filename.endswith('.npy'):
                dataset=np.load(filename)

            elif filename.endswith('.pkl'):
                dataset=pd.read_pickle(filename)            

            elif filename.endswith('.h5'):
                dataset=pd.read_hdf(filename) 

            elif filename.endswith('.hdf5'):
                dataset=pd.DataFrame(nparray(h5py.File((filename)))) 

        def browsemodel():

            global model
            global click
            click = 0
                    
            filename =tk.filedialog.askopenfilename(filetypes=(("pickle files","*.pkl"), ("sav files", "*.sav")))
            e_model.insert(tk.END, filename)
            model = pickle.load(open(filename, 'rb'))

        Button_entry = tk.Button(fen_test,text="Explorer",font=Browse_BUTTON ,command=browsedataset)
        Button_entry.config(fg=Browse_BUTTON_ft_color) #couleur du textelambda:
        Button_entry.config(bg=Browse_BUTTON_bg_color)
        Button_entry.grid(row=1,column=1,padx=5,pady=5)

        Button_emod = tk.Button(fen_test,text="Explorer",font=Browse_BUTTON ,command=browsemodel)
        Button_emod.config(fg=Browse_BUTTON_ft_color) #couleur du textelambda:
        Button_emod.config(bg=Browse_BUTTON_bg_color)
        Button_emod.grid(row=1,column=3,padx=5,pady=5)

        # conteneur entry 3, 4, 5, 6
        fen_test2 = tk.Frame(self, width=1000, height=100,)
        fen_test2.configure(bg=frame_color) #couleur de fond de la fenetre

        label=tk.Label(fen_test2,text='X train',font=BUTTON,justify=LEFT)
        label.config(fg=text_color) #couleur du texte
        label.config(background=frame_color) # couleur de fond du texte
        label.grid(row=0,column=0,padx=5,pady=5)         

        e_dataXtrain=tk.Entry(fen_test2,width=10 )
        e_dataXtrain.grid(row=0,column=1,padx=5,pady=5)  

        label=tk.Label(fen_test2,text='Y train',font=BUTTON,justify=LEFT)
        label.config(fg=text_color) #couleur du texte
        label.config(background=frame_color) # couleur de fond du texte
        label.grid(row=0,column=3,padx=5,pady=5)         

        e_dataYtrain=tk.Entry(fen_test2,width=10 )
        e_dataYtrain.grid(row=0,column=4,padx=5,pady=5)      

        label=tk.Label(fen_test2,text='X test',font=BUTTON,justify=LEFT)
        label.config(fg=text_color) #couleur du texte
        label.config(background=frame_color) # couleur de fond du texte
        label.grid(row=0,column=6,padx=5,pady=5)         

        e_dataXtest=tk.Entry(fen_test2,width=10 )
        e_dataXtest.grid(row=0,column=7,padx=5,pady=5)

        label=tk.Label(fen_test2,text='Y test',font=BUTTON,justify=LEFT)
        label.config(fg=text_color) #couleur du texte
        label.config(background=frame_color) # couleur de fond du texte
        label.grid(row=0,column=9,padx=5,pady=5)         

        e_dataYtest=tk.Entry(fen_test2,width=10 )
        e_dataYtest.grid(row=0,column=10,padx=5,pady=5)

        def browseX_train():
            
            global X_train
            global click
            click = 0
            
            filetrain = tk.filedialog.askopenfilename(filetypes=(("Numpy files","*.npy"),("CSV files","*.csv"),("pickle files","*.pkl"),("hdf5 files","*.hdf5"),("hdf5 files","*.h5")))
            e_dataXtrain.insert(tk.END, filetrain) # add this

            if filetrain.endswith('.csv'):
                X_train=pd.read_csv(filetrain)                    

            elif filetrain.endswith('.npy'):
                X_train=np.load(filetrain)
                
            elif filetrain.endswith('.pkl'):
                X_train=pd.read_pickle(filetrain) 
                               
            elif filetrain.endswith('.h5'):
                X_train=pd.read_hdf(filetrain) 
                               
            elif filetrain.endswith('.hdf5'):
                X_train=pd.DataFrame(nparray(h5py.File((filetrain))))

            if isinstance(X_train, np.ndarray):
                X_train = pd.DataFrame(X_train)

        def browseY_train():
            
            global Y_train
            global click
            click = 0
            
            filetrain = tk.filedialog.askopenfilename(filetypes=(("Numpy files","*.npy"),("CSV files","*.csv"),("pickle files","*.pkl"),("hdf5 files","*.hdf5"),("hdf5 files","*.h5")))
            e_dataYtrain.insert(tk.END, filetrain) # add this

            if filetrain.endswith('.csv'):
                Y_train=pd.read_csv(filetrain)            
                                       
            elif filetrain.endswith('.npy'):
                Y_train=np.load(filetrain)
                
            elif filetrain.endswith('.pkl'):
                Y_train=pd.read_pickle(filetrain) 
                               
            elif filetrain.endswith('.h5'):
                Y_train=pd.read_hdf(filetrain) 
                               
            elif filetrain.endswith('.hdf5'):
                Y_train=pd.DataFrame(nparray(h5py.File((filetrain)))) 
            
            if isinstance(Y_train, np.ndarray):
                Y_train = pd.DataFrame(Y_train)

        def browseX_test():
           
            global X_test
            global click
            click = 0
            
            filetest = tk.filedialog.askopenfilename(filetypes=(("Numpy files","*.npy"),("CSV files","*.csv"),("pickle files","*.pkl"),("hdf5 files","*.hdf5"),("hdf5 files","*.h5")))
            print(filetest.endswith)
            
            e_dataXtest.insert(tk.END, filetest) # add this
           
            if filetest.endswith('.csv'):
                X_test=pd.read_csv(filetest)
                                       
            elif filetest.endswith('.npy'):
                X_test=np.load(filetest)
                
            elif filetest.endswith('.pkl'):
                X_test=pd.read_pickle(filetest) 
                               
            elif filetest.endswith('.h5'):
                X_test=pd.read_hdf(filetest) 
                               
            elif filetest.endswith('.hdf5'):
                X_test=pd.DataFrame(nparray(h5py.File((filetest))))  
            
            if isinstance(X_test, np.ndarray):
                X_test = pd.DataFrame(X_test)

        def browseY_test():

            global Y_test
            global click
            click = 0
            
            filetest = tk.filedialog.askopenfilename(filetypes=(("Numpy files","*.npy"),("CSV files","*.csv"),("pickle files","*.pkl"),("hdf5 files","*.hdf5"),("hdf5 files","*.h5")))
            e_dataYtest.insert(tk.END, filetest) # add this
            if filetest.endswith('.csv'):
                Y_test=pd.read_csv(filetest)

            elif filetest.endswith('.npy'):
                Y_test=np.load(filetest)
                
            elif filetest.endswith('.pkl'):
                Y_test=pd.read_pickle(filetest) 
                               
            elif filetest.endswith('.h5'):
                Y_test=pd.read_hdf(filetest) 
                               
            elif filetest.endswith('.hdf5'):
                Y_test=pd.DataFrame(nparray(h5py.File((filetest)))) 
        
            if isinstance(Y_test, np.ndarray):
                Y_test = pd.DataFrame(Y_test)


        Button_etrain = tk.Button(fen_test2,text="Explorer",font=Browse_BUTTON ,command=browseX_train)
        Button_etrain.config(fg=Browse_BUTTON_ft_color) #couleur du texte lambda:
        Button_etrain.config(bg=Browse_BUTTON_bg_color)
        Button_etrain.grid(row=0,column=2,padx=5,pady=5)

        Button_etrain = tk.Button(fen_test2,text="Explorer",font=Browse_BUTTON ,command=browseY_train)
        Button_etrain.config(fg=Browse_BUTTON_ft_color) #couleur du texte lambda:
        Button_etrain.config(bg=Browse_BUTTON_bg_color)
        Button_etrain.grid(row=0,column=5,padx=5,pady=5)
        
        Button_etrain = tk.Button(fen_test2,text="Explorer",font=Browse_BUTTON ,command=browseX_test)
        Button_etrain.config(fg=Browse_BUTTON_ft_color) #couleur du texte lambda:
        Button_etrain.config(bg=Browse_BUTTON_bg_color)
        Button_etrain.grid(row=0,column=8,padx=5,pady=5)

        Button_etrain = tk.Button(fen_test2,text="Explorer",font=Browse_BUTTON ,command=browseY_test)
        Button_etrain.config(fg=Browse_BUTTON_ft_color) #couleur du texte lambda:
        Button_etrain.config(bg=Browse_BUTTON_bg_color)
        Button_etrain.grid(row=0,column=11,padx=5,pady=5)

        global currentGraph
        global currentdirectory

        #conteneur canvas   
        fen_can = tk.Frame(self, width=1000, height=100,)
        fen_can.configure(bg=frame_color) #couleur de fond de la fenetre

        canvas_PermImportance = tk.Canvas(fen_can,width=350, height=350, bg="white") 
        canvas_PermImportance.grid(row=0, column=0)
        
        def check():
            global dataset
            if ( (type(dataset)!=int) and (type(model)!=int) and (type(X_train) != int) and (type(X_test)!=int) and (type(Y_train) != int) and (type(Y_test) != int) ) :
                Permutation_Importance()

        def Permutation_Importance():
            global currentdirectory
            global X_train
            global X_test
            global model
            global Y_train
            global Y_test
            
            perm = PermutationImportance(model, random_state=1).fit(X_test, Y_test)
            explanation = eli5.explain.explain_weights(perm, feature_names = X_test.columns.tolist())
            df_explanation = format_as_dataframe(explanation)
            ax = df_explanation.plot()
            plt.xlabel('Feature')
            fig = ax.get_figure()
            fig.savefig(currentdirectory +'/PermutationImportance.png')
            plotPermutationImportance()

        def plotPermutationImportance(): 
		
            global currentGraph 
            global currentdirectory
            global click
            if (click==0) :
                # Clear all graphs drawn in figure 
                photo = tk.PhotoImage(file=currentdirectory + "/PermutationImportance.png")
                labelimg = tk.Label(canvas_PermImportance, image=photo)
                labelimg.image = photo
                labelimg.pack(pady=0, padx=0)
                click+=1


        # Create a tkinter button at the bottom of the window and link it with the updateGraph function 
        tk.Button(fen_can,text="Générer",command=check).grid(row=0, column=1) 

        # conteneur des boutons
        fen_boutons = tk.Frame(self, width=500, height=200,)
        fen_boutons.configure(bg=frame_color) #couleur de fond de la fenetre 

        button1=tk.Button(fen_boutons,text="Retour",font=BUTTON ,command=lambda:controller.show_frame(TabularPage))
        button1.config(fg=text_button) #couleur du texte
        button1.config(bg=bg_button)
        button1.place(relx=0,rely=1,anchor=tk.SW)

        button3=tk.Button(fen_boutons,text="Quitter",font=BUTTON ,command=lambda:controller.show_frame(PageQuit))
        button3.config(fg=text_button) #couleur du texte
        button3.config(bg=bg_button)
        button3.place(relx=1,rely=1,anchor=tk.SE)

        fen_test.pack(pady=2)
        fen_test2.pack(pady=2)
        fen_can.pack(pady=2)
        fen_boutons.pack(pady=30)
 
class SHAP (tk.Frame): # page qui demande confirmation avant de quitter

   def __init__(self,parent,controller):

        tk.Frame.__init__(self,parent)
        label=tk.Label(self,text="SHAP",font=LARGE_FONT)
        label.config(fg=text_color) #couleur du texte
        label.config(background=frame_color) # couleur de fond du texte
        label.pack(pady=25)

        # conteneur entry 1 et 2
        fen_test = tk.Frame(self, width=1000, height=100,)
        fen_test.configure(bg=frame_color) #couleur de fond de la fenetre

        label=tk.Label(fen_test,text='Data set',font=BUTTON,justify=LEFT)
        label.config(fg=text_color) #couleur du texte
        label.config(background=frame_color) # couleur de fond du texte
        label.grid(row=0,column=0,padx=5,pady=5)

        label=tk.Label(fen_test,text='Modèle',font=BUTTON,justify=LEFT)
        label.config(fg=text_color) #couleur du texte
        label.config(background=frame_color) # couleur de fond du texte
        label.grid(row=0,column=2,padx=5,pady=5)  

        e_dataset=tk.Entry(fen_test,width=50 )
        e_dataset.grid(row=1,column=0,padx=5,pady=5)

        e_model=tk.Entry(fen_test,width=50 )
        e_model.grid(row=1,column=2,padx=5,pady=5)

        global X_train
        global X_test
        global Y_train
        global Y_test
        global dataset
        global model
        global click
        
        click=0
        X_train, X_test, Y_train, Y_test, dataset, model = 0, 0, 0, 0, 0, 0

        def browsedataset():
            global dataset
            global click
            click = 0

            filename = tk.filedialog.askopenfilename(filetypes=(("Numpy files","*.npy"),("CSV files","*.csv"),("pickle files","*.pkl"),("hdf5 files","*.hdf5"),("hdf5 files","*.h5")))
            e_dataset.insert(tk.END, filename)
            if filename.endswith('.csv'):
                dataset=pd.read_csv(filename)
                 
            elif filename.endswith('.npy'):
                dataset=np.load(filename)

            elif filename.endswith('.pkl'):
                dataset=pd.read_pickle(filename)            

            elif filename.endswith('.h5'):
                dataset=pd.read_hdf(filename) 

            elif filename.endswith('.hdf5'):
                dataset=pd.DataFrame(nparray(h5py.File((filename)))) 

        def browsemodel():

            global model
            global click
            click = 0
                    
            filename =tk.filedialog.askopenfilename(filetypes=(("pickle files","*.pkl"), ("sav files", "*.sav")))
            e_model.insert(tk.END, filename)
            model = pickle.load(open(filename, 'rb'))

        Button_entry = tk.Button(fen_test,text="Explorer",font=Browse_BUTTON ,command=browsedataset)
        Button_entry.config(fg=Browse_BUTTON_ft_color) #couleur du textelambda:
        Button_entry.config(bg=Browse_BUTTON_bg_color)
        Button_entry.grid(row=1,column=1,padx=5,pady=5)

        Button_emod = tk.Button(fen_test,text="Explorer",font=Browse_BUTTON ,command=browsemodel)
        Button_emod.config(fg=Browse_BUTTON_ft_color) #couleur du textelambda:
        Button_emod.config(bg=Browse_BUTTON_bg_color)
        Button_emod.grid(row=1,column=3,padx=5,pady=5)

        # conteneur entry 3, 4, 5, 6
        fen_test2 = tk.Frame(self, width=1000, height=100,)
        fen_test2.configure(bg=frame_color) #couleur de fond de la fenetre

        label=tk.Label(fen_test2,text='X train',font=BUTTON,justify=LEFT)
        label.config(fg=text_color) #couleur du texte
        label.config(background=frame_color) # couleur de fond du texte
        label.grid(row=0,column=0,padx=5,pady=5)         

        e_dataXtrain=tk.Entry(fen_test2,width=10 )
        e_dataXtrain.grid(row=0,column=1,padx=5,pady=5)  

        label=tk.Label(fen_test2,text='Y train',font=BUTTON,justify=LEFT)
        label.config(fg=text_color) #couleur du texte
        label.config(background=frame_color) # couleur de fond du texte
        label.grid(row=0,column=3,padx=5,pady=5)         

        e_dataYtrain=tk.Entry(fen_test2,width=10 )
        e_dataYtrain.grid(row=0,column=4,padx=5,pady=5)      

        label=tk.Label(fen_test2,text='X test',font=BUTTON,justify=LEFT)
        label.config(fg=text_color) #couleur du texte
        label.config(background=frame_color) # couleur de fond du texte
        label.grid(row=0,column=6,padx=5,pady=5)         

        e_dataXtest=tk.Entry(fen_test2,width=10 )
        e_dataXtest.grid(row=0,column=7,padx=5,pady=5)

        label=tk.Label(fen_test2,text='Y test',font=BUTTON,justify=LEFT)
        label.config(fg=text_color) #couleur du texte
        label.config(background=frame_color) # couleur de fond du texte
        label.grid(row=0,column=9,padx=5,pady=5)         

        e_dataYtest=tk.Entry(fen_test2,width=10 )
        e_dataYtest.grid(row=0,column=10,padx=5,pady=5)

        def browseX_train():
            
            global X_train
            global click
            click = 0
            
            filetrain = tk.filedialog.askopenfilename(filetypes=(("Numpy files","*.npy"),("CSV files","*.csv"),("pickle files","*.pkl"),("hdf5 files","*.hdf5"),("hdf5 files","*.h5")))
            e_dataXtrain.insert(tk.END, filetrain) # add this

            if filetrain.endswith('.csv'):
                X_train=pd.read_csv(filetrain)                    

            elif filetrain.endswith('.npy'):
                X_train=np.load(filetrain)
                
            elif filetrain.endswith('.pkl'):
                X_train=pd.read_pickle(filetrain) 
                               
            elif filetrain.endswith('.h5'):
                X_train=pd.read_hdf(filetrain) 
                               
            elif filetrain.endswith('.hdf5'):
                X_train=pd.DataFrame(nparray(h5py.File((filetrain))))

            if isinstance(X_train, np.ndarray):
                X_train = pd.DataFrame(X_train)

        def browseY_train():
            
            global Y_train
            global click
            click = 0
            
            filetrain = tk.filedialog.askopenfilename(filetypes=(("Numpy files","*.npy"),("CSV files","*.csv"),("pickle files","*.pkl"),("hdf5 files","*.hdf5"),("hdf5 files","*.h5")))
            e_dataYtrain.insert(tk.END, filetrain) # add this

            if filetrain.endswith('.csv'):
                Y_train=pd.read_csv(filetrain)            
                                       
            elif filetrain.endswith('.npy'):
                Y_train=np.load(filetrain)
                
            elif filetrain.endswith('.pkl'):
                Y_train=pd.read_pickle(filetrain) 
                               
            elif filetrain.endswith('.h5'):
                Y_train=pd.read_hdf(filetrain) 
                               
            elif filetrain.endswith('.hdf5'):
                Y_train=pd.DataFrame(nparray(h5py.File((filetrain)))) 
            
            if isinstance(Y_train, np.ndarray):
                Y_train = pd.DataFrame(Y_train)

        def browseX_test():
           
            global X_test
            global click
            click = 0
            
            filetest = tk.filedialog.askopenfilename(filetypes=(("Numpy files","*.npy"),("CSV files","*.csv"),("pickle files","*.pkl"),("hdf5 files","*.hdf5"),("hdf5 files","*.h5")))
            print(filetest.endswith)
            
            e_dataXtest.insert(tk.END, filetest) # add this
           
            if filetest.endswith('.csv'):
                X_test=pd.read_csv(filetest)
                                       
            elif filetest.endswith('.npy'):
                X_test=np.load(filetest)
                
            elif filetest.endswith('.pkl'):
                X_test=pd.read_pickle(filetest) 
                               
            elif filetest.endswith('.h5'):
                X_test=pd.read_hdf(filetest) 
                               
            elif filetest.endswith('.hdf5'):
                X_test=pd.DataFrame(nparray(h5py.File((filetest))))  
            
            if isinstance(X_test, np.ndarray):
                X_test = pd.DataFrame(X_test)

        def browseY_test():

            global Y_test
            global click
            click = 0
            
            filetest = tk.filedialog.askopenfilename(filetypes=(("Numpy files","*.npy"),("CSV files","*.csv"),("pickle files","*.pkl"),("hdf5 files","*.hdf5"),("hdf5 files","*.h5")))
            e_dataYtest.insert(tk.END, filetest) # add this
            if filetest.endswith('.csv'):
                Y_test=pd.read_csv(filetest)

            elif filetest.endswith('.npy'):
                Y_test=np.load(filetest)
                
            elif filetest.endswith('.pkl'):
                Y_test=pd.read_pickle(filetest) 
                               
            elif filetest.endswith('.h5'):
                Y_test=pd.read_hdf(filetest) 
                               
            elif filetest.endswith('.hdf5'):
                Y_test=pd.DataFrame(nparray(h5py.File((filetest)))) 
        
            if isinstance(Y_test, np.ndarray):
                Y_test = pd.DataFrame(Y_test)


        Button_etrain = tk.Button(fen_test2,text="Explorer",font=Browse_BUTTON ,command=browseX_train)
        Button_etrain.config(fg=Browse_BUTTON_ft_color) #couleur du texte lambda:
        Button_etrain.config(bg=Browse_BUTTON_bg_color)
        Button_etrain.grid(row=0,column=2,padx=5,pady=5)

        Button_etrain = tk.Button(fen_test2,text="Explorer",font=Browse_BUTTON ,command=browseY_train)
        Button_etrain.config(fg=Browse_BUTTON_ft_color) #couleur du texte lambda:
        Button_etrain.config(bg=Browse_BUTTON_bg_color)
        Button_etrain.grid(row=0,column=5,padx=5,pady=5)
        
        Button_etrain = tk.Button(fen_test2,text="Explorer",font=Browse_BUTTON ,command=browseX_test)
        Button_etrain.config(fg=Browse_BUTTON_ft_color) #couleur du texte lambda:
        Button_etrain.config(bg=Browse_BUTTON_bg_color)
        Button_etrain.grid(row=0,column=8,padx=5,pady=5)

        Button_etrain = tk.Button(fen_test2,text="Explorer",font=Browse_BUTTON ,command=browseY_test)
        Button_etrain.config(fg=Browse_BUTTON_ft_color) #couleur du texte lambda:
        Button_etrain.config(bg=Browse_BUTTON_bg_color)
        Button_etrain.grid(row=0,column=11,padx=5,pady=5)

        global currentGraph
        global currentdirectory

        #conteneur canvas   
        fen_can = tk.Frame(self, width=1000, height=100,)
        fen_can.configure(bg=frame_color) #couleur de fond de la fenetre

        canvas_SHAP = tk.Canvas(fen_can,width=350, height=350, bg="white") 
        canvas_SHAP.grid(row=0, column=0)
        
        def check():
            global dataset
            if ( (type(dataset)!=int) and (type(model)!=int) and (type(X_train) != int) and (type(X_test)!=int) and (type(Y_train) != int) and (type(Y_test) != int) ) :
                SHAP ()

        def SHAP ():
            global currentdirectory
            global X_train
            global X_test
            global model
            global Y_train
            global Y_test
            
            shap_values = shap.TreeExplainer(model).shap_values(X_train, check_additivity=False)
            shap.summary_plot(shap_values, X_train, plot_type="bar", plot_size = (4, 4))
            plt.tight_layout()
            plt.savefig(currentdirectory + "/shap.png")
            plotSHAP()

        def plotSHAP(): 
		
            global currentGraph 
            global currentdirectory
            global click
            if (click==0) :
                # Clear all graphs drawn in figure 
                photo = tk.PhotoImage(file=currentdirectory + "/shap.png")
                labelimg = tk.Label(canvas_SHAP, image=photo)
                labelimg.image = photo
                labelimg.pack(pady=0, padx=0)
                click+=1


        # Create a tkinter button at the bottom of the window and link it with the updateGraph function 
        tk.Button(fen_can,text="Générer",command=check).grid(row=0, column=1) 

        # conteneur des boutons
        fen_boutons = tk.Frame(self, width=500, height=200,)
        fen_boutons.configure(bg=frame_color) #couleur de fond de la fenetre 

        button1=tk.Button(fen_boutons,text="Retour",font=BUTTON ,command=lambda:controller.show_frame(TabularPage))
        button1.config(fg=text_button) #couleur du texte
        button1.config(bg=bg_button)
        button1.place(relx=0,rely=1,anchor=tk.SW)

        button3=tk.Button(fen_boutons,text="Quitter",font=BUTTON ,command=lambda:controller.show_frame(PageQuit))
        button3.config(fg=text_button) #couleur du texte
        button3.config(bg=bg_button)
        button3.place(relx=1,rely=1,anchor=tk.SE)

        fen_test.pack(pady=2)
        fen_test2.pack(pady=2)
        fen_can.pack(pady=2)
        fen_boutons.pack(pady=30)
              
class Assistance (tk.Frame): 
    
    def __init__(self,parent,controller):
        
        tk.Frame.__init__(self,parent)
        
        # conteneur du titre
        fen_Title = tk.Frame(self, width=1000, height=250,)
        fen_Title.configure(bg=frame_color)#frame_color) #couleur de fond de la fenetre

        
        label=tk.Label(fen_Title,text="Découvrez la TOOL BOX\n",font=LARGE_FONT3)
        label.config(fg='tomato') #couleur du texte
        label.config(background=frame_color) # couleur de fond du texte
        label.pack(pady=10,padx=10)

        label=tk.Label(fen_Title,text=text_assistance,font=TEXT_Assistance,justify=LEFT) 
        label.config(fg=text_color) #couleur du texte
        label.config(background=frame_color) # couleur de fond du texte
        label.pack(pady=10,padx=10)

        
        # conteneur des boutons
        fen_boutons = tk.Frame(self, width=1000, height=600,)
        fen_boutons.configure(bg=frame_color) #couleur de fond de la fenetre 
        
        button1=tk.Button(fen_boutons,text="Retour",font=BUTTON ,command=lambda:controller.show_frame(StartPage2))
        button1.config(fg=text_button) #couleur du texte
        button1.config(bg=bg_button)
        button1.place(relx=0,rely=1,anchor=tk.SW)

        
        button3=tk.Button(fen_boutons,text="Quitter",font=BUTTON ,command=lambda:controller.show_frame(PageQuit))
        button3.config(fg=text_button) #couleur du texte
        button3.config(bg=bg_button)
        button3.place(relx=1,rely=1,anchor=tk.SE)
        
        fen_Title.pack(pady=20)
        fen_boutons.pack(pady=20)

class ExplanationPage (tk.Frame): 

    def __init__(self,parent,controller):
        
        tk.Frame.__init__(self,parent)
        label=tk.Label(self,text="Interprétez et comprenez avec TOOL BOX",font=LARGE_FONT)
        label.config(fg='tomato') #couleur du texte
        label.config(background=frame_color) # couleur de fond du texte
        label.pack(pady=100)
        
        
        label=tk.Label(self,text=text_explain,font=BUTTON,justify=LEFT) # affichage des règles du jeu de la vie
        label.config(background=frame_color) # couleur de fond du texte
        label.pack(pady=10,padx=10)

        # conteneur des boutons
        fen_boutons = tk.Frame(self, width=1000, height=600,)
        fen_boutons.configure(bg=frame_color) #couleur de fond de la fenetre 
        
        button1=tk.Button(fen_boutons,text="Continuer",font=BUTTON ,command=lambda:controller.show_frame(OptionPage))
        button1.config(fg=text_button) #couleur du texte
        button1.config(bg=bg_button)
        button1.place(relx=0,rely=1,anchor=tk.SW)
        
        button2=tk.Button(fen_boutons,text="Retour",font=BUTTON ,command=lambda:controller.show_frame(StartPage2))
        button2.config(fg=text_button) #couleur du texte
        button2.config(bg=bg_button)
        button2.place(relx=0.5,rely=1,anchor=tk.S)

        button3=tk.Button(fen_boutons,text="Quitter",font=BUTTON ,command=lambda:controller.show_frame(PageQuit))
        button3.config(fg=text_button) #couleur du texte
        button3.config(bg=bg_button)
        button3.place(relx=1,rely=1,anchor=tk.SE)

        fen_boutons.pack(pady=50)

class PageQuit (tk.Frame): # page qui demande confirmation avant de quitter
    
    def __init__(self,parent,controller):
        global currentdirectory
        tk.Frame.__init__(self,parent,)
        # conteneur des boutons
        fen = tk.Frame(self, width=1000, height=600,)
        fen.configure(bg=frame_color) #couleur de fond de la fenetre         
        # conteneur des boutons
        sub_fen = tk.Frame(fen, width=1000, height=300,)
        sub_fen.configure(bg=frame_color) #couleur de fond de la fenetre  
        
        photo = tk.PhotoImage(file=currentdirectory + "/logo.png")
        labelimg = tk.Label(fen, image=photo)
        labelimg.image = photo
        labelimg.grid(row=0,column=1)       
        
        label=tk.Label(sub_fen,text="Quitter la \t",font=LARGE_FONT)
        label.config(fg=text_color) #couleur du texte
        label.config(background=frame_color) #couleur de fond du texte
        label.grid(row=0,column=0)
        
        label=tk.Label(sub_fen,text="TOOLBOX \t",font=LARGE_FONT)
        label.config(fg='red') #couleur du texte
        label.config(background=frame_color) #couleur de fond du texte
        label.grid(row=1, column=0)
        
        sub_fen.grid(row=0,column=0)

        # conteneur des boutons
        fen_boutons = tk.Frame(self, width=1000, height=600,)
        fen_boutons.configure(bg=frame_color) #couleur de fond de la fenetre 

        button1=tk.Button(fen_boutons,text="Oui",font=BigBUTTON ,command=self.quit)
        button1.config(fg=text_button) #couleur du texte
        button1.config(background=bg_button) #couleur de fond du bouton
        button1.place(relx=0,rely=1,anchor=tk.SW)

        button2=tk.Button(fen_boutons,text="Non",font=BigBUTTON ,command=lambda:controller.show_frame(StartPage2))
        button2.config(fg=text_button) #couleur du texte
        button2.config(background=bg_button) #couleur de fond du bouton
        button2.place(relx=1,rely=1,anchor=tk.SE)
        

        fen.pack(pady=100)
        fen_boutons.pack(pady=50)


def main():

    app=XAI_Project()
    app.attributes('-fullscreen', True) # mode plein ecran
    app.mainloop()


if __name__ == '__main__':

    main()


