# PFE Description d'images en utilisant le Deep Learning
## Installation 
    Les bibliothèques python nécessaires pour applications sont notées dans le fichier all_lib.py
L’ajout des bibliothèques de TensorFlow Api manuellement se fasse :
  * En executant le fichier install_api.bat sous windows 
  * En compilant le projet [TensorFlow API](https://github.com/tensorflow/models/) 
par protoc et les ajoutant aux bibliothèques python 


## Utilisation 
Les codes dans le fichier [create dict data](https://github.com/A-RAMZI/PFE/tree/master/create%20data%20dict)  servent à extraire les données essentielles des fichiers Json du Dataset.
L'application permet de :
  * Parcourir le Dataset ( chaque partie est découpée en pages de 10 items et chaque item contient une image et les descriptions correspondantes)
  * Décrire les images selon notre modèle ( il suffit d'indiquer le chemin d'acces à l'image)
  * Comparer les descriptions du dataset avec la déscription générée si l'image appartient au Dataset)
    
![image info](./temp/appli.jpg)

## Test results
Les résultats Bleu-N sur les données de tests sont :
(0.3602033744470266, 0.1214666905397255, 0.03479153426583626, 0.010338026451352279)
