import tkinter as tk
from tkinter import messagebox

# Fonction pour effectuer la prédiction (pour l'exemple, on montre juste un message)
def predire():
    # Récupération des valeurs saisies par l'utilisateur
    age = entry_age.get()
    poids = entry_poids.get()
    taille = entry_taille.get()
    glucose = entry_glucose.get()
    tension = entry_tension.get()
    imc = entry_imc.get()
    insuline = entry_insuline.get()

    # Affiche un message d'exemple avec les valeurs saisies
    messagebox.showinfo("Résultat de la prédiction", f"Prédiction effectuée pour l'utilisateur avec les valeurs : \n"
                                                     f"Âge: {age}, Poids: {poids} kg, Taille: {taille} cm, Glucose: {glucose}, "
                                                     f"Tension: {tension}, IMC: {imc}, Insuline: {insuline}")

# Création de la fenêtre principale
root = tk.Tk()
root.title("DiaPredict - Prédiction des Risques de Diabète")
root.geometry("600x400")

# Création du titre
title_label = tk.Label(root, text="DiaPredict", font=("Helvetica", 24, "bold"), fg="purple")
title_label.pack(pady=10)

# Sous-titre
subtitle_label = tk.Label(root, text="Prédiction des risques de diabète", font=("Helvetica", 16))
subtitle_label.pack(pady=5)

# Cadre principal pour organiser les champs de saisie
frame = tk.Frame(root)
frame.pack(pady=20)

# Création des labels et champs de saisie
labels = [
    "Âge", "Poids (kg)", "Taille (cm)", "Taux de glucose", 
    "Tension artérielle", "Indice de Masse Corporelle (IMC)", "Insuline"
]

# Dictionnaire pour stocker les champs de saisie
entries = {}

# Création des champs
for label_text in labels:
    label = tk.Label(frame, text=label_text, font=("Helvetica", 12))
    label.pack(anchor="w")
    entry = tk.Entry(frame, font=("Helvetica", 12), width=30)
    entry.pack(pady=5)
    entries[label_text] = entry

# Assignation des champs à des variables pour un accès plus simple
entry_age = entries["Âge"]
entry_poids = entries["Poids (kg)"]
entry_taille = entries["Taille (cm)"]
entry_glucose = entries["Taux de glucose"]
entry_tension = entries["Tension artérielle"]
entry_imc = entries["Indice de Masse Corporelle (IMC)"]
entry_insuline = entries["Insuline"]

# Bouton prédire
predict_button = tk.Button(root, text="Prédire", command=predire, bg="lightgreen", font=("Helvetica", 12))
predict_button.pack(pady=20)

# Copyright en bas de la fenêtre
copyright_label = tk.Label(root, text="© 2023 Amine AIT MOKHTAR - DiaPredict - Tous droits réservés", font=("Helvetica", 10))
copyright_label.pack(side="bottom", pady=10)

# Lancement de la fenêtre
root.mainloop()
