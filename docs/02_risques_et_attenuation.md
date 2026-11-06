# 2. Analyse des risques et atténuation

Analyse des risques du projet de scoring churn (dataset Telco), avec signal
d'alerte, mesure d'atténuation et responsable désigné.

## 2.1 Matrice des risques

| Risque | Prob. | Impact | Signal d'alerte | Atténuation | Responsable |
|---|---|---|---|---|---|
| R1 — Fuite de données : feature construite à partir d'une information post-résiliation | Moyenne | Critique | Métriques de test nettement supérieures aux métriques de CV ; AUC quasi parfaite | Feature engineering entièrement dans le pipeline (`FunctionTransformer`) ; fit uniquement sur l'entraînement ; brouillon v0 rejeté explicitement | Data scientist |
| R2 — Dérive des données : les distributions évoluent (tarifs, produits) | Moyenne | Élevé | Baisse de ROC-AUC entre les campagnes | Suivi des scores par lot ; réentraînement trimestriel ; comparaison des distributions d'entrée | Data scientist |
| R3 — Seuil par défaut (0.5) incohérent avec le métier | Élevée | Élevé | Valeur nette négative sur les campagnes | Seuil optimisé économiquement (`optimize_threshold`) ; `BusinessContext` modifiable sans code | Sponsor |
| R4 — Coûts mal estimés (coût de churn, taux de sauvetage) | Moyenne | Élevé | Gain métier proche de zéro | Analyse de sensibilité du seuil ; validation des coûts réels par la finance | Sponsor + finance |
| R5 — Modèle perçu comme opaque, rejeté par les commerciaux | Moyenne | Moyen | La file d'appels est ignorée ; taux de contact en baisse | Explication SHAP par client (locale et globale) dans l'application ; session de sensibilisation de 30 minutes | Commercial |
| R6 — Écart entre l'environnement de démonstration et la production | Moyenne | Moyen | Erreurs d'import ou de packaging au déploiement | Déploiement Docker (`uv sync --no-dev --all-extras`) ; test de serveur local via `make serve` ; test de fumée Gradio dans le notebook | Tech |
| R7 — Perte ou dégradation du fichier de données source | Faible | Élevé | Échec du chargement, données incomplètes | Données montées en volume Docker ; contrôle qualité au chargement (dimensions, types, doublons) | Tech |
| R8 — Biais démographique (genre, citoyenneté senior) | Moyenne | Moyen | Écarts de précision et de rappel entre segments | Analyse par segment intégrée au notebook ; décision métier explicite sur les colonnes sensibles | Data scientist |
| R9 — Pollution du modèle par les effets de campagne (clients sauvés réétiquetés) | Moyenne | Moyen | Réentraînement dégradé après campagne | Historique des scores conservé ; réentraînement après un délai raisonnable | Data scientist |
| R10 — Budget et adoption : le projet s'arrête par manque de ROI démontré | Moyenne | Élevé | Le sponsor ne suit plus la démarche | Le calcul de ROI est présenté AVANT le déploiement (partie métier du notebook) ; livrable minimal dès les premières phases | Sponsor |

## 2.2 Mesures transversales

- Traçabilité : chaque transformation vit dans `src/churn_pred/telco.py` (pas de code
  disséminé dans le notebook) ; les artefacts enregistrés contiennent métriques et
  paramètres.
- Tests : 9 tests unitaires couvrent nettoyage, absence de fuite, théorie du seuil,
  entraînement et prédiction sur le bundle complet (`tests/test_telco.py`) ;
  `make test` les exécute en une commande.
- Réversibilité : le seuil, les coûts et le taux de sauvetage sont des paramètres de
  contexte métier ; si l'estimation change, aucun réentraînement n'est nécessaire.
- Livraison minimale d'abord : l'onglet « Score client » de l'application Gradio
  suffit pour soutenir une campagne ; les autres onglets (scoring de masse, carte
  du modèle) arrivent ensuite.

## 2.3 Risques acceptés et documentés

- Anomalie `TotalCharges` vide pour `tenure = 0` : mise à 0 après vérification
  manuelle, pour 11 nouveaux clients. Pas d'imputation au-delà : un client qui vient
  d'arriver n'a pas encore généré de facture.
- Absence de données d'usage (logs de consommation) : les features comportementales
  sont construites à partir des colonnes disponibles uniquement ; une version
  suivante pourra les enrichir sans casser l'interface (`input_contract`).
