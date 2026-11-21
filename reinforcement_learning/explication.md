# ViZDoom:

**Les biblio à installer :**

**vizdoom**
**numpy**
**Pillow**
**torch**

## 1- Choix du l'algo et la justification :

Nous avons choisi le Deep Q-Network car notre agent doit apprendre à agir à partir des images de ViZDoom. Cet espace d’observation est trop vaste pour utiliser efficacement une table de Q-Learning classique. Le DQN approxime les valeurs des actions à l’aide d’un réseau de neurones et convient à notre ensemble d’actions discrètes : se déplacer à gauche, à droite et tirer. Il permet d’apprendre par interaction avec le jeu, sans disposer d’un modèle explicite de ses transitions. Ce choix demande toutefois davantage de calcul et de réglages qu’une méthode tabulaire.

## 2- PLAN DU PROJET:

first step : Environment ViZDoom & préparation du jeu

second step: Agent DQN et algo d'apprentissage

third step: Entraînement , évaluation et analyse

more details about each step :

### ~1~ **Environment ViZDoom & préparation du jeu: ( ALL DONE)**

**Objectif: permettre à l'agent d'interagir facilement au jeu**

- Configurer le scenario choisi: actions autorisées, durée maximale d’une partie et récompenses.

- Écrire une fonction pour exécuter une action et récupérer :

  -> la nouvelle observation

  -> la récompense

  -> l’indication que la partie est terminée

- Préparer les images : réduction de taille, passage éventuel en niveaux de gris et normalisation des pixels

- Gérer correctement la fin d’une partie, quand aucune nouvelle image n’est disponible.

- Créer un agent aléatoire pour vérifier que l’environnement fonctionne et fournir une première référence.

- Permettre d’activer ou de masquer la fenêtre du jeu.

### ~2~ **Agent DQN et algorithme d’apprentissage: (ALL DONE)**

**Objectif: créer l’agent qui apprend à choisir de meilleures actions**

- Construire le réseau de neurones : il reçoit l’observation et produit une valeur estimée pour chaque action. **( DONE)**

- Implémenter le choix d’action **epsilon-greedy** : **( DONE)**

  -> parfois choisir au hasard pour explorer ;

  -> sinon choisir l’action ayant la meilleure valeur estimée.

- Faire diminuer progressivement le taux d’exploration. **( DONE)**

- Créer une mémoire d’expériences, contenant :"observation, action, récompense, observation suivante, fin de partie". **( DONE)**

- Tirer des lots d’expériences dans cette mémoire pour entraîner le réseau. **( DONE)**

- Calculer les cibles d’apprentissage en tenant compte des fins de partie. **( DONE)**

- Utiliser un réseau cible, mis à jour périodiquement, pour stabiliser l’apprentissage. **( DONE)**

- Ajouter la sauvegarde et le chargement des poids du modèle. **( DONE)**

- Tester les fonctionnalités de l'agent DQN avec des expériences fictives. **( DONE)**

### ~3~ **Entraînement, évaluation et analyse**

**Objectif: organiser les expériences et montrer si l’agent progresse**

- Écrire la boucle d’entraînement qui relie l’environnement de l'étape 1 à l'agent fait en etape 2 :

  1. Commencer une partie.
  2. Demander une action à l’agent.
  3. Exécuter cette action dans le jeu.
  4. Enregistrer l’expérience et entraîner l’agent.
  5. Recommencer jusqu’à la fin de la partie.

- Centraliser les paramètres : nombre d’épisodes, taux d’apprentissage, taille des lots, exploration, etc.

- Enregistrer les résultats dans un CSV : récompense, réussite, durée et nombre de tirs si disponibles.

- Sauvegarder régulièrement le modèle pendant l’entraînement.

- Écrire une évaluation séparée, sans apprentissage et avec une exploration désactivée.

- Comparer l’agent entraîné à l’agent aléatoire sur les mêmes conditions de test.

- Produire les graphiques :

  -> récompense moyenne au cours de l’entraînement ;

  -> taux de réussite ;

  -> comparaison avant/après.

- Préparer un lancement de démonstration avec la fenêtre du jeu visible.

## 3-Explication:

### ~1~ **Environment ViZDoom & préparation du jeu:**

**Configuration du jeu:**

Nous avons créé le dossier **reinforcement_learning** pour notre code Python. Dans **environment.py**, la classe VizDoomEnv charge le scénario basic, configure l’affichage et permet d’afficher ou de masquer la fenêtre. La durée maximale est fixée à 300 tics, avec une pénalité de −1 par tic pour encourager l’agent à agir rapidement.

**Récompenses du scénario:**

Le scénario basic utilise les récompenses suivantes pour encourager l’agent à éliminer la cible rapidement en limitant les tirs inutiles :

Dans le **scénario basic**, les récompenses prévues sont :

Événement                       Récompense

Éliminer la cible                 +106

Chaque tir effectué              −5

Chaque tic de jeu écoulé          −1

**Définition des actions:**

Nous avons défini trois actions : aller à gauche, aller à droite et tirer. Chacune correspond à un numéro, respectivement 0, 1 et 2, que le futur agent DQN pourra choisir.

**Interaction avec l’environnement:**

La méthode reset() commence une nouvelle partie. La méthode step(action) exécute une action et retourne l’observation suivante, la récompense et l’indication de fin de partie. Elle vérifie aussi que l’action est valide et que la partie n’est pas déjà terminée. Enfin, close() ferme le jeu.

**Préparation des images:**

Les images reçues sont converties en niveaux de gris, réduites à 84 × 84 pixels et normalisées entre 0 et 1. Chaque observation est un tableau de forme (1, 84, 84) et de type float32, prêt à être utilisé par le futur réseau de neurones.

**Test avec un agent aléatoire:**

Dans **random_agent.py**, nous avons créé un agent qui choisit ses actions au hasard et joue plusieurs parties. Ce test vérifie que l’environnement fonctionne et fournit une référence pour comparer ensuite les performances du DQN. Aucun apprentissage n’a encore lieu.

**Enregistrement des résultats:**

Après chaque partie, nous ajoutons au même fichier CSV la date d’exécution, le numéro de la partie et sa récompense totale. Les anciens résultats sont conservés pour suivre les différentes exécutions.

### ~2~ **Agent DQN et algorithme d’apprentissage:**

**Création du réseau:**

Nous avons créé dans **model.py** un réseau de neurones avec **PyTorch**. Il reçoit une image en niveaux de gris de 84 × 84 pixels.

Puis, on fait une **Extraction des caractéristiques**, trois couches de convolution extraient progressivement les caractéristiques visuelles de l’image. Les fonctions ReLU permettent au réseau d’apprendre des relations non linéaires.

après, **le calcul des valeurs Q**, les caractéristiques sont aplaties en un vecteur, puis traitées par des couches linéaires. Le réseau produit trois valeurs Q, correspondant aux actions : gauche, droite et tirer. Elles estiment les récompenses futures attendues ; ce ne sont pas des probabilités.

et enfin, la **Vérification du fonctionnement**, nous avons testé le réseau avec une image remplie de zéros. L’entrée a la forme (1, 1, 84, 84) et la sortie (1, 3), ce qui confirme que les dimensions sont correctes. Le réseau n’est pas encore entraîné : ses paramètres sont initialisés au hasard.

**Création de l'agent DQN:**

Nous avons créé dans **agent.py** la classe DQNAgent qui utilise le réseau de neurones de model.py pour choisir les actions et apprendre à partir des expériences du jeu.

L'agent possède un réseau principal, un réseau cible et plusieurs paramètres pour contrôler l'apprentissage.

**Choix d'action epsilon-greedy:**

Nous avons implémenté dans **agent.py** la méthode choisir_action() qui utilise la stratégie epsilon-greedy.

L'agent choisit parfois une action au hasard pour explorer le jeu. Sinon, il utilise le réseau de neurones pour calculer les valeurs Q et choisir l'action avec la plus grande valeur grâce à argmax().

Au début, epsilon est fixé à 1.0, ce qui signifie que l'agent explore à 100 %.

**Diminution du taux d'exploration:**

Nous avons ajouté la méthode diminuer_epsilon() qui permet de réduire progressivement l'exploration.

On multiplie epsilon par un facteur de 0.995 après chaque appel de la méthode, sans descendre en dessous de 0.05.

Cela permet à l'agent d'explorer beaucoup au début, puis d'utiliser davantage ce qu'il apprend.

**Création de la mémoire d'expériences:**

Nous avons créé dans **replay_memory.py** la classe ReplayMemory pour conserver les expériences de l'agent.

Chaque expérience contient cinq informations :

- state : l'observation avant l'action.
- action : l'action choisie.
- reward : la récompense reçue.
- next_state : l'observation suivante.
- done : indique si la partie est terminée.

Nous avons utilisé deque pour stocker jusqu'à 10000 expériences. Lorsque la mémoire est pleine, les plus anciennes sont supprimées automatiquement.

**Sélection des mini-batchs:**

Dans replay_memory.py, nous avons créé la méthode echantillonner() qui utilise random.sample() pour sélectionner des expériences au hasard.

Dans agent.py, la méthode apprendre() récupère par défaut un lot de 32 expériences pour entraîner le réseau.

Cela permet d'apprendre à partir de plusieurs situations différentes.

**Calcul des valeurs Q et apprentissage:**

Nous avons ajouté dans **agent.py** la méthode apprendre() qui permet au réseau de modifier ses poids pour améliorer ses estimations.

D'abord, nous sélectionnons des expériences dans la mémoire, puis nous utilisons le réseau principal pour calculer les valeurs Q prédites.

Ensuite, nous calculons les valeurs Q cibles avec l'équation de Bellman :

Q_cible = récompense + gamma × meilleure valeur Q suivante

Le paramètre gamma est fixé à 0.99 pour prendre en compte les récompenses futures.

Lorsque la partie est terminée, on ne prend pas en compte les récompenses futures.

Après, nous calculons l'erreur entre les valeurs prédites et les valeurs cibles avec SmoothL1Loss.

Enfin, nous utilisons l'optimiseur Adam avec un taux d'apprentissage de 0.0001 pour modifier les poids du réseau et réduire l'erreur.

**Utilisation du réseau cible:**

Nous avons créé un deuxième réseau de neurones appelé **target_model**.

Le réseau principal est utilisé pour choisir les actions et apprendre, tandis que le réseau cible sert à calculer les valeurs Q futures.

Au début, les deux réseaux possèdent les mêmes poids.

Nous avons ajouté la méthode mettre_a_jour_cible() qui copie les poids du réseau principal vers le réseau cible toutes les 1000 mises à jour d'apprentissage.

Cela permet de stabiliser l'apprentissage, car les valeurs cibles ne changent pas à chaque modification du réseau principal.

**Sauvegarde et chargement du modèle:**

Nous avons ajouté deux méthodes dans **agent.py** :

- sauvegarder() : enregistre le modèle dans un fichier .pth avec torch.save().
- charger() : récupère les paramètres sauvegardés avec torch.load().

La sauvegarde contient les poids du réseau principal, les poids du réseau cible, l'état de l'optimiseur, epsilon et le nombre de mises à jour d'apprentissage.

Cela permet de conserver le modèle et de réutiliser les paramètres appris sans repartir de zéro.

**Tests de fonctionnement:**

Nous avons créé le fichier **test_agent.py** pour vérifier le fonctionnement de notre agent DQN.

Nous avons généré 40 expériences fictives pour tester l'apprentissage sans lancer ViZDoom.

Les tests vérifient :

- L'initialisation des deux réseaux avec les mêmes poids.
- L'enregistrement des expériences dans la mémoire.
- Le calcul de l'erreur pendant l'apprentissage.
- La synchronisation du réseau cible.
- La sauvegarde et le chargement des paramètres.

**Résultats obtenus:**

    Initialisation des deux réseaux : OK
    Expériences enregistrées : 40
    Première mise à jour - perte : 0.5269865393638611
    Synchronisation du réseau cible : OK
    Sauvegarde : OK
    Chargement : OK
    Poids et paramètres restaurés : OK
    Tous les tests sont réussis !

Ces tests montrent que les fonctionnalités de l'agent fonctionnent avec des expériences fictives. L'entraînement réel dans ViZDoom sera réalisé dans la partie 3.

## QLQ PETITES PRECISIONS:

- Les observations ont la forme (1, 84, 84), en float32, entre 0 et 1.
- Les actions sont 0 = gauche, 1 = droite, 2 = tirer.
- En fin de partie, step() retourne observation = None.
- Le réseau produit 3 valeurs Q, mais il n’a pas encore été entraîné sur de vraies parties ViZDoom.
- L'agent utilise epsilon-greedy pour choisir ses actions.
- Epsilon commence à 1.0 et diminue progressivement jusqu'à 0.05 lorsque diminuer_epsilon() est appelée.
- La Replay Memory conserve jusqu'à 10000 expériences.
- L'apprentissage utilise par défaut des mini-batchs de 32 expériences.
- Le réseau cible est synchronisé toutes les 1000 mises à jour.
- Le modèle peut être sauvegardé et chargé avec un fichier .pth.
- Les tests de la partie 2 ont été exécutés avec succès.