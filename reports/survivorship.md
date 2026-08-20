# Survivorship bias — quantification et direction

Genere par `src/universe.py`. Source : `reports/identity_validation.csv`.

## 1. Ce que le point-in-time corrige

L'appartenance a l'indice est lue sur la page Wikipedia **telle qu'elle existait**
a chaque date de selection (25 photos semestrielles, lag median 2 jours). Le screener
ne voit donc jamais un titre avant son entree dans l'indice, ni apres sa sortie.

Cas le plus parlant : **PCG (PG&E)** est hors indice de 2019-07 a 2022-07 —
Chapter 11 apres le Camp Fire, retrait, puis reintegration. Un univers bati sur les
constituants **actuels** inclurait PCG et le traderait a travers sa faillite,
en sachant qu'il en est ressorti. Le point-in-time l'exclut automatiquement.

## 2. Ce qui reste non corrige

Union brute : **38 tickers**. Univers tradable : **33**.
Exclus : **5** (GAS, NU, POM, TE, TEG).

| Ticker | Snapshots membres | Statut | Motif |
|---|---|---|---|
| `GAS` | 6 | NO_DATA | AGL Resources, rachete par Southern Company (2016). Aucune donnee Yahoo : perte seche. |
| `NU` | 3 | RECYCLED | Northeast Utilities, RENOMME Eversource (ES) en 2015. Ticker repris depuis 2021 par Nu Holdings (neobanque bresilienne). Perte apparente seulement : la serie ES remonte a 2014 et porte l'historique. |
| `POM` | 5 | RECYCLED | Pepco Holdings, rachete par Exelon (2016). Ticker recycle en 2025. |
| `TE` | 6 | RECYCLED | TECO Energy, rachete par Emera (2016). Ticker recycle en 2020. |
| `TEG` | 4 | RECYCLED | Integrys Energy, rachete par WEC (2015). Ticker recycle fin 2015. |

Toutes ces exclusions sont des **sorties d'indice par rachat** (aucune faillite).

## 3. Magnitude

- Slots d'appartenance (snapshot x ticker) : 732 bruts, 708 conserves → **3.3 % perdus**.
- La perte est concentree sur 2014-2019 ; a partir de 2019-07 elle est nulle.

| Snapshot | Secteur (brut) | Tradable | Perdus | Paires testables |
|---|---|---|---|---|
| 2014-01 | 31 | 26 | 5 | 465 → 325 |
| 2014-07 | 30 | 25 | 5 | 435 → 300 |
| 2015-01 | 30 | 25 | 5 | 435 → 300 |
| 2015-07 | 30 | 26 | 4 | 435 → 325 |
| 2016-01 | 29 | 26 | 3 | 406 → 325 |
| 2016-07 | 29 | 27 | 2 | 406 → 351 |

## 4. Direction du biais

Les cinq noms perdus sont sortis de l'indice **par acquisition**. Or une OPA est
precisement l'evenement qui casse une relation de cointegration de la facon la plus
violente : le prix de la cible saute au prix d'offre puis se fige, definitivement
decorrele de son secteur. Une position short sur la cible au moment de l'annonce
subit une perte instantanee et non-mean-reverting.

**Consequence : leur exclusion SURESTIME la performance de la strategie.** Le biais
va dans le sens favorable, ce qui est le cas le plus dangereux — il ne se signale
pas de lui-meme.

**Borne d'ordre de grandeur.** La perte est entierement concentree sur les six
premiers snapshots (2014-01 a 2016-07) et **nulle a partir de 2017-01**. Au pire
snapshot (2014-01), 5 noms manquent sur 31, soit 16 % de l'univers et
1-(26/31)^2 ≈ **30 % des paires potentielles** non testables. A partir de 2017 le
survivorship residuel sur la composition est nul.

Consequence pratique pour le walk-forward : le premier fold (selection 2014-2016)
est le plus expose. Une option honnete est de reporter son resultat separement
plutot que de l'agreger silencieusement aux autres.

## 5. Limites de la reconstruction elle-meme

- Wikipedia est edite avec un lag de quelques jours (median 2j, max 19j ici). Sans
  effet a une cadence de re-selection annuelle.
- **EVRG** (Evergy, ne en 2018 de la fusion Westar/Great Plains) a un historique Yahoo
  remontant a 2014 : ce sont les prix du predecesseur. Legitime, mais a noter.
- Le controle d'identite repose sur le recouvrement entre cotation et appartenance.
  Il attraperait moins bien un recyclage de ticker survenant *pendant* la fenetre
  d'appartenance — cas non observe ici.