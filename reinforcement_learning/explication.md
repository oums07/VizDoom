ViZDoom:





**Les biblio à installer :**

**vizdoom**

**numpy**

**Pillow**

**torch**





1- Choix du l'algo et la justification :



Nous avons choisi le Deep Q-Network  car notre agent doit apprendre à agir à partir des images de ViZDoom. Cet espace d’observation est trop vaste pour utiliser efficacement une table de Q-Learning classique. Le DQN approxime les valeurs des actions à l’aide d’un réseau de neurones et convient à notre ensemble d’actions discrètes : se déplacer à gauche, à droite et tirer. Il permet d’apprendre par interaction avec le jeu, sans disposer d’un modèle explicite de ses transitions. Ce choix demande toutefois davantage de calcul et de réglages qu’une méthode tabulaire.





2- PLAN DU PROJET: 



first step : Environment ViZDoom \& préparation du jeu 

second step: Agent DQN et algo d'apprentissage

third step: Entraînement , évaluation et analyse



more details about each step :



&#x20;  \~1\~ **Environment ViZDoom \& préparation du jeu: ( ALL DONE)**

**Objectif: permettre à l'agent d'interagir facilement au jeu** 

&#x20;**-** Configurer le scenario choisi: actions autorisées, durée maximale d’une partie et récompenses.

&#x20;**-** Écrire une fonction pour exécuter une action et récupérer :

&#x20;  -> la nouvelle observation 

&#x20;  -> la récompense 

&#x20;  -> l’indication que la partie est terminée

&#x20;**-** Préparer les images : réduction de taille, passage éventuel en niveaux de gris et normalisation des pixels

&#x20;**-** Gérer correctement la fin d’une partie, quand aucune nouvelle image n’est disponible.

&#x20;**-** Créer un agent aléatoire pour vérifier que l’environnement fonctionne et fournir une première référence.

&#x20;**-** Permettre d’activer ou de masquer la fenêtre du jeu.



&#x20;     \~2\~ **Agent DQN et algorithme d’apprentissage:**

**Objectif: créer l’agent qui apprend à choisir de meilleures actions**



\-Construire le réseau de neurones : il reçoit l’observation et produit une valeur estimée pour chaque action. **( DONE)**

\-Implémenter le choix d’action **epsilon-greedy** :

&#x20; -> parfois choisir au hasard pour explorer ;

&#x20; -> sinon choisir l’action ayant la meilleure valeur estimée.

\-Faire diminuer progressivement le taux d’exploration.

\-Créer une mémoire d’expériences, contenant :"observation, action, récompense, observation suivante, fin de partie".

\-Tirer des lots d’expériences dans cette mémoire pour entraîner le réseau.

\-Calculer les cibles d’apprentissage en tenant compte des fins de partie.

\-Utiliser un réseau cible, mis à jour périodiquement, pour stabiliser l’apprentissage.

\-Ajouter la sauvegarde et le chargement des poids du modèle.



&#x20;  \~3\~ **Entraînement, évaluation et analyse**

**Objectif: organiser les expériences et montrer si l’agent progresse**

\-Écrire la boucle d’entraînement qui relie l’environnement de l'étape 1 à l'agent fait en etape 2 :

&#x20;  1. Commencer une partie.

&#x20;  2. Demander une action à l’agent.

&#x20;  3. Exécuter cette action dans le jeu.

&#x20;  4. Enregistrer l’expérience et entraîner l’agent.

&#x20;  5. Recommencer jusqu’à la fin de la partie.



\-Centraliser les paramètres : nombre d’épisodes, taux d’apprentissage, taille des lots, exploration, etc.

\-Enregistrer les résultats dans un CSV : récompense, réussite, durée et nombre de tirs si disponibles.

\-Sauvegarder régulièrement le modèle pendant l’entraînement.

\-Écrire une évaluation séparée, sans apprentissage et avec une exploration désactivée.

\-Comparer l’agent entraîné à l’agent aléatoire sur les mêmes conditions de test.

\-Produire les graphiques :

&#x20;  -> récompense moyenne au cours de l’entraînement ;

&#x20;  -> taux de réussite ;

&#x20;  -> comparaison avant/après.

\-Préparer un lancement de démonstration avec la fenêtre du jeu visible.





3-Explication:



&#x20;   \~1\~ **Environment ViZDoom \& préparation du jeu:**

**Configuration du jeu:**

&#x20; Nous avons créé le dossier **reinforcement\_learning** pour notre code Python. Dans **environment.py**, la classe VizDoomEnv charge le scénario basic, configure l’affichage et permet d’afficher ou de masquer la fenêtre. La durée maximale est fixée à 300 tics, avec une pénalité de −1 par tic pour encourager l’agent à agir rapidement.



**Récompenses du scénario:**

&#x20;Le scénario basic utilise les récompenses suivantes pour encourager l’agent à éliminer la cible rapidement en limitant les tirs inutiles :

Dans le **scénario basic**, les récompenses prévues sont :

Événement	                    Récompense

Éliminer la cible	              +106

Chaque tir effectué	               −5

Chaque tic de jeu écoulé	       −1



**Définition des actions:**

&#x20;Nous avons défini trois actions : aller à gauche, aller à droite et tirer. Chacune correspond à un numéro, respectivement 0, 1 et 2, que le futur agent DQN pourra choisir.



**Interaction avec l’environnement:**

&#x20;La méthode reset() commence une nouvelle partie. La méthode step(action) exécute une action et retourne l’observation suivante, la récompense et l’indication de fin de partie. Elle vérifie aussi que l’action est valide et que la partie n’est pas déjà terminée. Enfin, close() ferme le jeu.



**Préparation des images:**

&#x20;Les images reçues sont converties en niveaux de gris, réduites à 84 × 84 pixels et normalisées entre 0 et 1. Chaque observation est un tableau de forme (1, 84, 84) et de type float32, prêt à être utilisé par le futur réseau de neurones.

&#x20;

**Test avec un agent aléatoire:**

&#x20;Dans **random\_agent.py**, nous avons créé un agent qui choisit ses actions au hasard et joue plusieurs parties. Ce test vérifie que l’environnement fonctionne et fournit une référence pour comparer ensuite les performances du DQN. Aucun apprentissage n’a encore lieu.



**Enregistrement des résultats:**

&#x20;Après chaque partie, nous ajoutons au même fichier CSV la date d’exécution, le numéro de la partie et sa récompense totale. Les anciens résultats sont conservés pour suivre les différentes exécutions.



&#x20; \~2\~ **Agent DQN et algorithme d’apprentissage:**

**Création du réseau:**

Nous avons créé dans **model.py** un réseau de neurones avec **PyTorch**. Il reçoit une image en niveaux de gris de 84 × 84 pixels.

Puis, on fait une **Extraction des caractéristiques ,**trois couches de convolution extraient progressivement les caractéristiques visuelles de l’image. Les fonctions ReLU permettent au réseau d’apprendre des relations non linéaires.

après, **le calcul des valeurs Q**, les caractéristiques sont aplaties en un vecteur, puis traitées par des couches linéaires. Le réseau produit trois valeurs Q, correspondant aux actions : gauche, droite et tirer. Elles estiment les récompenses futures attendues ; ce ne sont pas des probabilités.

et enfin , la **Vérification du fonctionnement**, nous avons testé le réseau avec une image remplie de zéros. L’entrée a la forme (1, 1, 84, 84) et la sortie (1, 3), ce qui confirme que les dimensions sont correctes. Le réseau n’est pas encore entraîné : ses paramètres sont initialisés au hasard.





QLQ PETITES PRECISIONS:

* Les observations ont la forme (1, 84, 84), en float32, entre 0 et 1.
* Les actions sont 0 = gauche, 1 = droite, 2 = tirer.
* En fin de partie, step() retourne observation = None.
* Le réseau produit 3 valeurs Q, mais il n’est pas encore entraîné.













