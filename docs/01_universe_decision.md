# Phase 1 — Décision d'univers (FIGÉE le 2026-08-19, avant tout backtest)

> Ce document est écrit **avant** la première ligne de code de backtest et n'est pas
> révisable à la lumière des résultats. Toute révision ultérieure serait du data
> snooping au niveau de l'univers et devrait être signalée comme telle.

## 1. Univers : Utilities du S&P 500 (GICS 55)

**Décision.** Actions du secteur GICS *Utilities* ayant appartenu au S&P 500 à un
moment quelconque entre 2014-01-01 et 2026-06-30. Environ 31 noms courants, ~35-40
noms en union sur la période.

**Pourquoi intra-secteur.** La cointégration n'est pas une régularité statistique
gratuite : elle exige que deux prix partagent la *même* tendance stochastique, de
sorte qu'une combinaison linéaire soit I(0). Un prior économique est donc nécessaire.
Intra-secteur il est identifiable : même exposition aux taux longs, même régime
réglementaire (rendement autorisé sur base d'actifs), mêmes coûts d'input.
Bénéfice secondaire, non négligeable : cela réduit le nombre de tests de
N(N-1)/2 sur ~500 noms (124 750) à ~465, donc allège massivement la charge de
correction pour tests multiples (Phase 3).

**Pourquoi les utilities spécifiquement.**
- Facteur commun le plus net de tout le marché actions US (duration + réglementation)
  → proportion attendue de vraies paires cointégrées non nulle, ce qui rend
  l'expérience « sélection naïve vs corrigée » informative plutôt que dégénérée.
- N = 31 → 465 paires → sous H0 à α = 5 %, **~23 découvertes fallacieuses attendues**.
  Chiffre concret, opposable au nombre de paires effectivement « significatives ».
- Trois ruptures de régime datables **et explicables économiquement** :
  1. COVID (mars 2020) — choc de liquidité, corrélations → 1.
  2. Choc de taux (2022) — dispersion des sensibilités de duration.
  3. Repricing IA / datacenters (2023-2024) — les producteurs *merchant*
     (VST, CEG, NRG) décrochent structurellement des régulés (ED, WEC, ...).
     C'est LA rupture de cointégration à raconter en entretien.

**Objection anticipée.** « Ton spread n'est qu'un pari de duration déguisé. »
Réponse : partiellement vrai par construction, et c'est mesurable — le ratio de
couverture absorbe l'exposition commune aux taux ; le résidu ne doit plus être
expliqué par les taux. À vérifier explicitement (Phase 7 : régression du P&L sur
les variations de taux 10 ans).

**Alternatives écartées et pourquoi.**
| Alternative | Raison de l'écarter |
|---|---|
| Banques régionales (KRE) | Multiple testing plus riche (~4 000 paires) et survivorship spectaculaire (SVB/SBNY/FRC à zéro), mais les prix des délistés sont introuvables gratuitement → la plomberie de données aurait mangé le budget méthodologique. |
| Énergie E&P (XOP) | Vague de M&A 2023-24 qui tronque de nombreuses séries ; hétérogénéité de levier → cointégration fragile. |
| Semi-conducteurs | Trop tendanciel ; risque réel de ne trouver aucune paire cointégrée, ce qui viderait V2 de son contenu. |

## 2. Période : 2014-01-01 → 2026-06-30

~12,5 ans. Découpage walk-forward visé : **3 ans in-sample (sélection) / 1 an
out-of-sample (trading)**, fenêtre glissante d'un an → **~9 folds**.

Compromis assumé : remonter avant 2014 aurait ajouté des folds mais aggravé le
survivorship bias et fait porter le test sur une structure sectorielle
(pré-boom renouvelables, pré-spin-off CEG) peu comparable à l'actuelle.

## 3. Prix : total-return (ajustés splits + dividendes)

**Décision.** `auto_adjust=True` (yfinance), c'est-à-dire des prix ajustés des
splits *et* des dividendes.

**Pourquoi c'est une décision et pas un détail.** Le rendement du dividende des
utilities est de 3-4 % par an et **très dispersé** entre les noms. Sur prix bruts,
la différence de rendement entre deux titres injecte une dérive quasi-déterministe
dans le spread : le résidu n'est plus stationnaire et on rejetterait des paires
authentiquement cointégrées. Sur prix total-return, cette dérive disparaît. C'est
aussi le bon objet économique : la P&L d'une position long/short encaisse les
dividendes.

**Fuite résiduelle assumée.** Les prix ajustés sont *rétro*-ajustés par les
dividendes futurs : la série de 2018 telle que téléchargée en 2026 n'est pas celle
qu'un trader observait en 2018. L'effet est d'ordre 2 sur un spread long/short
(les deux jambes sont ajustées dans le même sens) mais il existe et il est nommé ici.

## 4. Survivorship bias

**Correctif appliqué.** Appartenance au S&P 500 reconstruite *point-in-time* à
partir de la table historique des ajouts/retraits. À chaque date de sélection, le
screener ne voit que les noms membres de l'indice **à cette date**. Cela élimine la
composante « choix des candidats » du biais.

**Biais résiduel, non corrigé, et sa direction.** Les prix des sociétés délistées
(faillite, rachat) ne sont pas disponibles gratuitement. Les noms qui disparaissent
sont majoritairement des noms en difficulté ou rachetés — c'est-à-dire précisément
des cas où une relation de cointégration casse violemment (rachat = découplage
instantané du prix vers le prix d'offre). Les exclure **surestime** la performance
de la stratégie. Magnitude à quantifier en Phase 7 : nombre de sorties d'indice
dans le secteur sur la période, rapporté au nombre de paires actives.

**Limite de la reconstruction.** Le secteur GICS listé sur la source publique est
le secteur *courant* ; pour les noms sortis de l'indice avant aujourd'hui, le
secteur est réattribué manuellement (liste courte, vérifiable, versionnée dans
`src/universe.py`).

## 5. Filtre de liquidité — causal par construction

Seuil de dollar-volume médian appliqué **fold par fold, sur la fenêtre in-sample
uniquement**. Calculer ce filtre sur l'échantillon complet serait une fuite : on
sélectionnerait les titres qui *resteront* liquides, ce qui est une information
future, corrélée à la survie et donc à la performance.

## 6. Marché unique, devise unique

Tous les titres cotent aux US, même clôture, même devise. On évite le piège des
paires trans-marchés où l'asynchronicité des heures de clôture fabrique une
prédictibilité illusoire du spread.
