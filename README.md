# Statistical arbitrage — sélection honnête sous tests multiples & conscience du régime

*English version: [README.en.md](README.en.md). Les commentaires du code source sont en anglais (public visé : fonds anglophones).*

Backtest d'une stratégie de *pairs trading* sur les utilities du S&P 500, construite
autour d'une question : **où, pourquoi et quand la stratégie se dégrade** — pas
autour d'un P&L.

Trois axes :
- **V1 — Régime** : la cointégration n'est pas stable ; détecter les ruptures et sortir.
- **V2 — Sélection honnête** : screener des centaines de paires est un problème de
  tests multiples ; corriger (Bonferroni / Benjamini-Hochberg) et séparer strictement
  sélection *in-sample* et trading *out-of-sample*.
- **V3 — Paniers** (extension) : généralisation multivariée via Johansen.

## État d'avancement

| Phase | Contenu | Statut |
|---|---|---|
| 0 | Setup, seed, structure | fait |
| 1 | Données & univers point-in-time | **fait** |
| 2 | Fondations cointégration | **fait — anglais** |
| 3 | V2 — sélection sous tests multiples | **fait — anglais** |
| 4 | V1 — conscience du régime | **fait — anglais** |
| 5 | V3 — paniers Johansen | **fait — anglais** |
| 6-8 | Backtest, analyse, rédaction | **fait — anglais** |
| 3 | V2 — sélection sous tests multiples | à venir |

## Phase 1 — ce qui est établi

**Univers figé avant tout backtest** (`docs/01_universe_decision.md`) : utilities du
S&P 500 (GICS 55), 2014-01 → 2026-06, prix total-return. Le document est daté et
non révisable a posteriori — changer d'univers après avoir vu un P&L serait du data
snooping au niveau du secteur.

**Appartenance point-in-time reconstruite** à partir de l'historique des révisions
Wikipédia : à chaque date de sélection, on lit la page *telle qu'elle existait*.
25 photos semestrielles, lag médian de 2 jours. Cela donne l'appartenance **et** le
secteur GICS observés à l'époque, y compris pour des sociétés aujourd'hui disparues.

Deux résultats que cette construction fait apparaître et qu'un univers « constituants
actuels » masquerait totalement :

1. **PCG (PG&E) est hors indice de 2019-07 à 2022-07** — Chapter 11 après le Camp
   Fire. Un univers naïf le traderait à travers sa faillite, en sachant qu'il en est
   ressorti vivant.
2. **Quatre tickers sont recyclés.** `NU` désignait Northeast Utilities jusqu'en 2015 ;
   depuis décembre 2021 il désigne Nu Holdings, une néobanque brésilienne. Idem pour
   `POM`, `TE`, `TEG`. Un téléchargement naïf par ticker injecte les prix d'une société
   sans rapport dans l'univers — **silencieusement**. Le contrôle d'identité
   (`validate_identity`) est donc bloquant : l'historique de prix doit recouvrir la
   fenêtre d'appartenance.

**Survivorship résiduel quantifié** (`reports/survivorship.md`) : 33 tickers tradables
sur 38 en union ; 3,3 % des slots d'appartenance perdus, concentrés sur 2014-2016 et
**nuls à partir de 2017**. Les cinq noms perdus sont tous des sorties **par
acquisition** — l'événement qui casse une cointégration de la façon la plus violente.
Leur exclusion **surestime** donc la performance : le biais va dans le sens favorable,
le cas le plus dangereux.

## Correction apportée par la Phase 2

La Phase 1 testait la cointégration avec `adfuller` sur le résidu OLS. **C'est le mauvais jeu de
valeurs critiques** : le résidu est *estimé*, donc sa distribution est décalée. Sous H₀ (marches
aléatoires indépendantes, n=750), `adfuller` rejette **14,6 %** du temps au seuil de 5 %, contre
4,2 % pour `coint` avec les valeurs d'Engle-Granger — un facteur **3,5** sur les faux positifs,
avant même le multiple testing. Sur le screen 2016-2018, le nombre de paires « significatives »
passe de 112 à **47**. La Phase 2 utilise `coint` partout ; les notebooks de Phase 1 sont laissés
tels quels, comme trace de ce qui a été fait.

Les PDF anglais contiennent en outre une section **fondements mathématiques** (définitions,
théorèmes, démonstrations courtes) que la version française du PDF de Phase 1 n'a pas.

**À partir de la Phase 2, les livrables sont en anglais uniquement** — voir
[docs/02_cointegration_decisions.en.md](docs/02_cointegration_decisions.en.md) et
[notebooks/02_cointegration_foundations_EN.ipynb](notebooks/02_cointegration_foundations_EN.ipynb).

## Documents

- **[reports/Phase1_decisions_expliquees.pdf](reports/Phase1_decisions_expliquees.pdf)** (EN : [Phase1_decisions_explained.pdf](reports/Phase1_decisions_explained.pdf)) —
  chaque décision de la Phase 1 expliquée en détail : alternatives, arbitrages, pièges,
  réponses aux objections d'entretien, glossaire.
- **[notebooks/01_phase1_univers_et_donnees.ipynb](notebooks/01_phase1_univers_et_donnees.ipynb)** (EN : [01_phase1_universe_and_data_EN.ipynb](notebooks/01_phase1_universe_and_data_EN.ipynb)) —
  le récit exécuté, graphiques compris, y compris les deux fois où l'intuition de départ n'a pas
  survécu au test (forward-fill, prix bruts).

## Deux résultats obtenus en testant nos propres hypothèses

1. **Le forward-fill ne fabrique pas de retour à la moyenne.** L'argument standard est faux :
   sur marche aléatoire pure, le taux de rejet ADF est inchangé (6,8 % → 6,5 %). Le vrai défaut
   est la tradabilité — ~3,8 % des signaux tombent un jour sans cotation.
2. **Les prix bruts changent la sélection des paires.** Sur 465 paires en 2016-2018, les deux
   séries divergent sur 11 %. L'essentiel est du bruit de p-value près du seuil, mais
   l'asymétrie des basculements francs — **16 paires fabriquées contre 1 détruite** — est robuste.

## Garde-fous méthodologiques

- Split strictement temporel ; sélection figée après l'in-sample.
- Aucun forward-fill des prix : non pas parce qu'il fabriquerait du retour à la
  moyenne (claim testé et réfuté, cf. notebook §5), mais parce qu'il génère des
  signaux les jours sans cotation — ~3,8 % des entrées à un prix inexistant.
- Filtre de liquidité recalculé fold par fold sur la fenêtre in-sample uniquement.
- Prix total-return : le dividende des utilities (3-4 %/an, très dispersé) injecterait
  sinon une dérive déterministe dans le spread.
- Coûts de transaction inclus dès le premier backtest ; exécution décalée à t+1.
- Seed fixé (`src/config.py`), caches disque pour la reproductibilité.

## Structure

```
src/config.py      constantes figées (univers, période, seed)
src/universe.py    appartenance point-in-time + contrôle d'identité des tickers
src/data.py        téléchargement, nettoyage non destructif, diagnostic de couverture
docs/              décisions méthodologiques datées
reports/           quantification du survivorship, validation d'identité
tests/             tests unitaires
```

## Installation

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m src.universe        # construit l'univers point-in-time
.venv/bin/python -m pytest tests/ -q
```
