# Assistant documentaire RAG

Application permettant de poser des questions sur plusieurs PDF
et d’obtenir des réponses en français avec les sources et les pages.

## Fonctionnalités

- Extraction et nettoyage du texte des PDF.
- Découpage du texte en passages.
- Recherche sémantique avec des embeddings.
- Recherche par mots-clés avec BM25.
- Traduction anglaise des questions pour compléter la recherche.
- Fusion des classements et reclassement par CrossEncoder.
- Génération locale des réponses avec Qwen via Ollama.
- Affichage des passages utilisés et des numéros de page.

## Technologies

Python, Streamlit, PyMuPDF, NumPy, Sentence Transformers,
BM25, CrossEncoder, Ollama et Qwen.

## Installation

Prérequis : Python et Ollama installés.

Créer et activer un environnement virtuel :

    python -m venv .venv
    source .venv/bin/activate

Installer les dépendances :

    python -m pip install -r requirements.txt

Télécharger le modèle de génération :

    ollama pull qwen3:4b-instruct

Le service Ollama doit être accessible sur :
http://127.0.0.1:11434

Les modèles d’embeddings et de reclassement sont téléchargés
automatiquement lors de leur première utilisation.

## Lancement

    source .venv/bin/activate
    python -m streamlit run app.py

Ouvrir http://localhost:8501 dans le navigateur.

## Utilisation

1. Charger un ou plusieurs PDF contenant du texte sélectionnable.
2. Cocher « PDF à deux colonnes » si nécessaire.
3. Poser une question précise sur les documents.
4. Vérifier la réponse à l’aide des passages cités.

Le réglage des colonnes s’applique à tous les PDF chargés.

## Premiers tests manuels

- LoFTR : un passage pertinent sur le raffinement sous-pixel
  a été retrouvé après reclassement.
- RAG : la réponse sur RAG-Sequence et RAG-Token reste
  partiellement correcte, avec des simplifications techniques.
- Information absente : l’assistant a correctement refusé
  de déduire la carte graphique de l’utilisateur.

Ces premiers essais ne constituent pas une évaluation complète.

## Limites

- OCR non intégrée : les PDF scannés ne sont pas pris en charge.
- Extraction parfois incorrecte des tableaux, figures et formules.
- Possibilité de manquer des passages pertinents.
- Possibilité d’erreurs dans les réponses et les citations.
- Temps de traitement variable sur processeur.
- Questions traitées indépendamment, sans mémoire conversationnelle.

## Exécution locale

Après téléchargement des modèles, le traitement des documents
et la génération des réponses s’exécutent localement.
