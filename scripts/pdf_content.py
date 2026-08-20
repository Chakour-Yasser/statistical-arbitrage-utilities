# -*- coding: utf-8 -*-
"""Contenu du document explicatif de la Phase 1.

Les chiffres sont recalcules depuis les donnees reelles : le PDF ne peut pas
diverger du code.
"""
from __future__ import annotations
import pandas as pd
from src import config as C
from src.universe import validate_identity, tradable_universe


def _facts():
    memb = pd.read_parquet(C.DATA_PROC / "membership_utilities.parquet")
    close = pd.read_parquet(C.DATA_PROC / "prices_adj.parquet")
    val = validate_identity(memb, close)
    tu = tradable_universe(memb, close)
    return memb, close, val, tu


def blocks() -> list:
    memb, close, val, tu = _facts()
    n_union = memb.shape[1]
    n_trad = tu.shape[1]
    n_last = int(tu.iloc[-1].sum())
    n_max = int(tu.sum(axis=1).max())
    n_min = int(tu.sum(axis=1).min())
    pairs_max = n_max * (n_max - 1) // 2
    fp_max = 0.05 * pairs_max
    slots_all, slots_kept = int(memb.values.sum()), int(tu.values.sum())
    lost = sorted(set(memb.columns) - set(tu.columns))
    B = []
    A = B.append

    # ====================================================================== #
    A(("h1", "1. À quoi sert ce document", False))
    A(("p", "Ce document reprend **chaque décision** prise pendant la Phase 1 du projet "
           "et l'explique en détail : ce qu'on a décidé, pourquoi, quelles étaient les "
           "alternatives, et ce qu'on perd en choisissant. Il est écrit pour que tu "
           "puisses défendre le projet au tableau sans notes."))
    A(("p", "La Phase 1 ne produit aucun signal, aucun backtest, aucun P&L. Elle produit "
           "uniquement un **univers de titres et une série de prix**. C'est pour cette "
           "raison qu'elle paraît anodine — et c'est exactement pour cette raison qu'elle "
           "est dangereuse : une erreur commise ici ne provoque jamais de message "
           "d'erreur. Elle produit un backtest qui tourne, qui affiche un Sharpe, et qui "
           "est faux. Toutes les phases suivantes héritent silencieusement de ses défauts."))
    A(("key", "La règle qui structure toute la Phase 1",
       ["Un backtest ne peut utiliser, à la date *t*, que de l'information réellement "
        "disponible à la date *t*. Toute violation de cette règle s'appelle une **fuite** "
        "(*look-ahead bias*), et une fuite gonfle la performance sans jamais se signaler.",
        "La difficulté est que les fuites les plus graves ne se cachent pas dans le "
        "signal. Elles se cachent dans la construction des données — c'est-à-dire ici."]))
    A(("p", "Le document se lit dans l'ordre. Les sections 2 et 3 posent le vocabulaire "
           "(à sauter si tu es à l'aise avec la cointégration). Les sections 4 à 9 "
           "détaillent les décisions. La section 10 répond aux deux questions posées à la "
           "fin de la Phase 1. La section 11 est un glossaire."))

    # ====================================================================== #
    A(("h1", "2. Le vocabulaire, sans raccourci", True))
    A(("h2", "2.1 Ce qu'est une paire"))
    A(("p", "Une *paire* est un couple d'actions (A, B) qu'on trade **ensemble et en sens "
           "opposés** : on achète A et on vend B à découvert, ou l'inverse. On ne parie "
           "jamais sur la direction du marché, mais sur **l'écart** entre les deux titres."))
    A(("p", "Cet écart s'appelle le **spread**. Sa définition la plus simple :"))
    A(("math", "s(t) = log P<sub>A</sub>(t) − β · log P<sub>B</sub>(t)"))
    A(("p", "Le coefficient β s'appelle le **ratio de couverture** (*hedge ratio*). Il "
           "répond à la question : « pour un dollar investi en A, combien faut-il vendre "
           "de B pour neutraliser le risque commun ? » Si β = 1,3, on vend 1,3 dollar de B "
           "pour chaque dollar de A."))
    A(("p", "On travaille sur les **log-prix** et non sur les prix bruts pour deux raisons. "
           "D'abord, la différence de log-prix est une quantité sans dimension "
           "(un log-ratio), donc comparable entre paires de niveaux de prix très "
           "différents. Ensuite, les rendements sont additifs en log, ce qui rend le "
           "calcul du P&L direct."))
    A(("h2", "2.2 La stratégie en une phrase"))
    A(("p", "Si le spread est **stationnaire** — c'est-à-dire s'il oscille autour d'une "
           "moyenne fixe au lieu de partir à la dérive — alors quand il s'écarte "
           "anormalement de cette moyenne, on parie qu'il y reviendra. On vend le spread "
           "quand il est haut, on l'achète quand il est bas, et on ferme au retour à la "
           "moyenne."))
    A(("p", "Toute la stratégie repose donc sur **une seule hypothèse** : la stationnarité "
           "du spread. La Phase 2 la testera, la Phase 3 corrigera le fait qu'on la teste "
           "des centaines de fois, et la Phase 4 traitera le fait qu'elle cesse d'être "
           "vraie sans prévenir."))

    A(("h2", "2.3 Corrélation et cointégration : la distinction qui tombe en entretien"))
    A(("p", "C'est **la** question de base sur ce projet. Deux séries peuvent être "
           "fortement corrélées sans être cointégrées, et inversement."))
    A(("p", "La **corrélation** porte sur les *rendements*, c'est-à-dire sur les variations "
           "de court terme. Dire que A et B sont corrélés à 0,9 signifie que leurs "
           "mouvements quotidiens vont dans le même sens. Cela ne dit **rien** sur "
           "l'écart de leurs niveaux à long terme : deux titres peuvent monter et "
           "descendre ensemble tous les jours tout en s'éloignant indéfiniment l'un de "
           "l'autre."))
    A(("p", "La **cointégration** porte sur les *niveaux*. Elle dit qu'il existe un β tel "
           "que la combinaison log P<sub>A</sub> − β log P<sub>B</sub> soit stationnaire, "
           "alors que log P<sub>A</sub> et log P<sub>B</sub> pris séparément ne le sont pas."))
    A(("p", "Formellement : un prix d'action se comporte comme une série **intégrée d'ordre 1**, "
           "notée I(1) — sa différence (le rendement) est stationnaire, mais son niveau "
           "dérive sans borne. La cointégration est le cas remarquable où une combinaison "
           "linéaire de deux séries I(1) est I(0), c'est-à-dire stationnaire."))
    A(("key", "L'intuition à donner au tableau",
       ["Deux séries I(1) cointégrées partagent la **même tendance stochastique**. "
        "Le β est le poids exact qui l'élimine par soustraction : ce qui reste est le "
        "bruit stationnaire autour de la relation d'équilibre.",
        "L'image classique : un homme ivre et son chien. Chacun marche au hasard "
        "(chaque trajectoire est I(1), imprévisible). Mais la laisse borne leur "
        "distance : cette distance est stationnaire. La corrélation décrit s'ils font "
        "le même pas au même instant ; la cointégration décrit l'existence de la laisse."]))
    A(("p", "Conséquence directe pour ce projet : la cointégration n'est pas une "
           "coïncidence statistique, elle exige un **mécanisme économique** qui joue le "
           "rôle de la laisse. C'est cet argument, et lui seul, qui justifie la décision "
           "de la section 4."))

    # ====================================================================== #
    A(("h1", "3. Pourquoi la Phase 1 mérite un document entier", True))
    A(("p", "Le pairs trading a une réputation détestable auprès des fonds quantitatifs, "
           "et pour une bonne raison : c'est l'exercice où il est le plus facile de "
           "produire un backtest spectaculaire et entièrement faux. Trois mécanismes y "
           "suffisent, et tous les trois se jouent en Phase 1 ou juste après."))
    A(("ol", [
        "**Le survivorship bias.** On construit l'univers avec les sociétés qui existent "
        "aujourd'hui, donc on ne trade que des survivants.",
        "**Le multiple testing.** On teste des centaines de paires et on garde les "
        "meilleures, sans corriger le fait qu'en testant assez, on trouve toujours.",
        "**Le look-ahead d'exécution ou de normalisation.** On utilise, pour décider à la "
        "date *t*, une statistique calculée sur des données postérieures à *t*.",
    ]))
    A(("p", "Le point 2 est traité en Phase 3, le point 3 en Phases 2 et 6. **Le point 1 "
           "se joue entièrement en Phase 1**, et c'est le seul qu'on ne peut plus corriger "
           "après coup : si l'univers est mal construit, tout ce qui suit est contaminé."))
    A(("warn", "Ce qui rend ces erreurs particulières",
       ["Un bug informatique classique s'annonce : le programme s'arrête, une exception "
        "est levée, un test échoue. Les erreurs de Phase 1 ne font rien de tout cela.",
        "Elles produisent un backtest qui s'exécute normalement, des courbes lisses et un "
        "ratio de Sharpe présentable. La seule façon de les détecter est de les chercher "
        "délibérément, en écrivant des contrôles dont l'unique fonction est de les faire "
        "échouer."]))

    # ====================================================================== #
    A(("h1", "4. Décision n°1 — L'univers : les utilities du S&P 500", True))
    A(("h2", "4.1 Première sous-décision : rester à l'intérieur d'un seul secteur"))
    A(("p", "On aurait pu former les paires sur l'ensemble du S&P 500. On a choisi de "
           "restreindre à un secteur GICS unique. Deux justifications, de nature différente."))
    A(("h3", "Justification économique"))
    A(("p", "La section 2.3 a établi que la cointégration exige une tendance stochastique "
           "commune. À l'intérieur d'un secteur, on peut la nommer : pour les utilities, "
           "c'est l'exposition aux taux longs (ce sont des actifs à duration très élevée, "
           "achetés pour leur rendement), le régime réglementaire (les tarifs sont fixés "
           "par des commissions sur une base d'actifs) et les coûts d'input (gaz, "
           "combustibles). Entre deux secteurs, ce mécanisme n'existe pas — une "
           "cointégration entre une banque et un producteur de semi-conducteurs serait "
           "un artefact d'échantillon."))
    A(("h3", "Justification statistique"))
    A(("p", "Le nombre de paires croît en N². Sur 500 titres, on aurait "
           "500 × 499 / 2 = 124 750 tests à corriger. Avec une correction de Bonferroni, "
           "le seuil deviendrait α/124750 ≈ 4 × 10<sup>-7</sup> : plus aucune paire ne passerait, "
           "et la Phase 3 n'aurait rien à montrer. Restreindre le secteur ramène le "
           f"problème à une échelle où la correction est **contraignante mais pas "
           f"destructrice** — {pairs_max} tests."))
    A(("p", "C'est un point important pour l'entretien : la restriction sectorielle n'est "
           "pas une facilité destinée à réduire le travail. C'est une **réduction de "
           "l'espace de recherche fondée sur un prior**, et c'est précisément ce que fait "
           "un chercheur quantitatif sérieux avant de lancer un screener."))

    A(("h2", "4.2 Deuxième sous-décision : lequel des onze secteurs"))
    A(("p", "Quatre critères ont été posés **avant** de regarder les données, ce qui est "
           "l'élément décisif : choisir le secteur après avoir vu quel secteur donne le "
           "meilleur P&L serait du multiple testing au niveau du secteur, et détruirait "
           "l'argument même du projet."))
    A(("table",
       ["Critère", "Ce qu'il exige", "Pourquoi"],
       [["Facteur commun identifiable",
         "Pouvoir nommer le mécanisme économique qui lie les titres",
         "Sans lui, la cointégration détectée est un artefact d'échantillon"],
        ["Taille suffisante",
         "Au moins ~25 titres, soit ≥ 300 paires",
         "En dessous, la démonstration sur les tests multiples n'a plus de poids"],
        ["Rupture datable et explicable",
         "Au moins un épisode où la relation casse, dont on connaît la cause",
         "La V1 doit expliquer *pourquoi* ça casse, pas seulement constater"],
        ["Homogénéité d'exécution",
         "Un seul marché, une seule devise, une seule heure de clôture",
         "Évite la prédictibilité illusoire due à l'asynchronicité des clôtures"]],
       [0.24, 0.36, 0.40],
       "Les quatre critères de sélection du secteur, figés avant tout examen des données."))
    A(("p", "Les utilities satisfont les quatre. Le facteur commun est le plus net de tout "
           "le marché actions américain. Le secteur compte entre "
           f"{n_min} et {n_max} titres selon la date. Et il offre non pas une mais **trois "
           "ruptures racontables** :"))
    A(("ul", [
        "**Mars 2020, COVID.** Choc de liquidité généralisé : toutes les corrélations "
        "convergent vers 1, les relations de long terme sont temporairement noyées.",
        "**2022, choc de taux.** La remontée rapide des taux frappe les utilities via "
        "leur duration, mais **de façon inégale** selon le niveau d'endettement et la "
        "structure de dette de chaque société. La dispersion des sensibilités casse des "
        "relations qui tenaient depuis des années.",
        "**2023-2024, repricing IA.** La demande électrique des centres de données "
        "revalorise brutalement les producteurs *merchant* — ceux qui vendent au prix de "
        "marché (VST, CEG, NRG) — tandis que les régulés (ED, WEC) restent plafonnés par "
        "leur tarification. C'est une rupture **structurelle** : les deux groupes cessent "
        "définitivement de partager la même tendance.",
    ]))
    A(("key", "La rupture de 2023-2024 est le cœur narratif de la V1",
       ["Elle a tout ce qu'un interviewer veut entendre : une date, un mécanisme "
        "économique précis, une conséquence statistique vérifiable (le β dérive, la "
        "demi-vie explose, la p-value de l'ADF remonte), et un caractère **permanent** — "
        "ce n'est pas un choc temporaire dont il faut attendre la fin, c'est une "
        "relation qui n'existe plus et dont il faut sortir."]))

    A(("h2", "4.3 Ce qu'on a écarté, et pourquoi"))
    A(("table",
       ["Secteur", "Ce qu'il apportait", "Pourquoi écarté"],
       [["Banques régionales (KRE)",
         "~90-140 noms, donc ~4 000 paires : tests multiples bien plus spectaculaires. "
         "Et mars 2023 (SVB, Signature, First Republic) fait du survivorship bias le "
         "sujet central plutôt qu'une note de bas de page.",
         "Les prix des sociétés en faillite sont introuvables gratuitement. Le budget "
         "serait passé en plomberie de données au lieu de méthodologie."],
        ["Énergie E&P (XOP)",
         "Facteur commun évident (le prix du pétrole) et rupture de 2020 spectaculaire, "
         "avec un WTI négatif en avril.",
         "La vague de fusions-acquisitions de 2023-2024 tronque un grand nombre de "
         "séries. L'hétérogénéité des niveaux d'endettement fragilise la cointégration."],
        ["Semi-conducteurs",
         "La rupture IA de 2023 est extrêmement visible.",
         "Secteur trop tendanciel : risque réel de ne trouver quasiment aucune paire "
         "cointégrée, ce qui viderait la Phase 3 de tout contenu."]],
       [0.20, 0.42, 0.38]))

    A(("h2", "4.4 L'objection que tu vas recevoir"))
    A(("warn", "« Ton spread n'est qu'un pari de duration déguisé »",
       ["L'objection est sérieuse et partiellement fondée : si tous les titres réagissent "
        "aux taux, un long/short entre deux d'entre eux reste exposé au différentiel de "
        "sensibilité aux taux.",
        "**La bonne réponse n'est pas de nier, c'est de mesurer.** Le β de couverture "
        "absorbe par construction l'exposition *commune*. Ce qui doit être vérifié, c'est "
        "que le **résidu** — donc le P&L — n'est plus explicable par les taux. Le test "
        "est explicite : régresser les rendements quotidiens de la stratégie sur les "
        "variations du taux 10 ans américain et montrer que le coefficient n'est pas "
        "significatif. C'est prévu en Phase 7.",
        "Répondre « je ne sais pas, mais voici le test que je ferais » vaut infiniment "
        "mieux que de nier l'objection."]))

    # ====================================================================== #
    A(("h1", "5. Décision n°2 — La période et le découpage temporel", True))
    A(("h2", "5.1 Pourquoi 2014 → 2026"))
    A(("p", "Le choix de la période résulte d'un arbitrage entre deux forces opposées."))
    A(("p", "**Allonger** l'historique augmente le nombre de sous-périodes indépendantes, "
           "donc la fiabilité des conclusions : avec trois ans de données on ne peut rien "
           "dire de la stabilité d'une stratégie. **Raccourcir** l'historique améliore la "
           "qualité des données (le survivorship s'aggrave en remontant) et la pertinence "
           "(la structure du secteur en 2010 — avant l'essor des renouvelables, avant la "
           "scission de Constellation — ressemble peu à celle d'aujourd'hui)."))
    A(("p", "Douze ans et demi place le curseur là où les deux effets s'équilibrent : assez "
           "de sous-périodes pour parler de dégradation du signal, pas assez pour tester "
           "une stratégie sur un secteur qui n'existe plus."))
    A(("h2", "5.2 Le walk-forward : trois ans de sélection, un an de trading"))
    A(("p", "C'est le protocole qui garantit que la sélection des paires et leur trading "
           "n'ont jamais lieu sur les mêmes données."))
    A(("code",
       "fold 1 :  SELECTION 2014-2016 (3 ans)  ->  TRADING 2017 (1 an)\n"
       "fold 2 :  SELECTION 2015-2017          ->  TRADING 2018\n"
       "fold 3 :  SELECTION 2016-2018          ->  TRADING 2019\n"
       "...\n"
       "fold 9 :  SELECTION 2022-2024          ->  TRADING 2025\n"
       "\n"
       "A chaque fold : univers point-in-time recalcule, filtre de liquidite\n"
       "recalcule, cointegration retestee, paires selectionnees -- puis FIGEES.\n"
       "La fenetre de trading n'influence JAMAIS la selection qui la precede."))
    A(("p", "Pourquoi trois ans de sélection ? Un test de cointégration a besoin de "
           "suffisamment de points pour distinguer une vraie relation d'équilibre d'un "
           "hasard : environ 750 jours de bourse. Sur un an, le test de Dickey-Fuller "
           "augmenté n'a quasiment aucune puissance — il ne rejette presque jamais "
           "l'hypothèse de non-stationnarité, même quand elle est fausse."))
    A(("p", "Pourquoi un an de trading ? Au-delà, on suppose implicitement que la "
           "relation sélectionnée en 2016 tient encore en 2020. C'est exactement "
           "l'hypothèse que la V1 conteste. Un an est le compromis entre la fréquence de "
           "re-sélection et le coût de rotation."))
    A(("warn", "Le piège dans lequel il ne faut pas tomber",
       ["Il est tentant, une fois arrivé en période de trading, de re-tester la "
        "cointégration et de ne garder que les paires qui « fonctionnent encore ». "
        "C'est une **réintroduction directe du look-ahead** : on utiliserait "
        "l'information de la période de trading pour décider quoi trader pendant cette "
        "même période.",
        "La règle est absolue : **la sélection est figée à la fin de l'in-sample.** "
        "Si une paire se dégrade pendant le trading, c'est le mécanisme de détection de "
        "régime (V1) qui doit la sortir — et ce mécanisme ne regarde, lui aussi, que le "
        "passé."]))

    # ====================================================================== #
    A(("h1", "6. Décision n°3 — Quels prix exactement", True))
    A(("h2", "6.1 Prix bruts, ajustés des splits, ou total-return ?"))
    A(("p", "Trois séries de prix différentes existent pour une même action, et le choix "
           "entre elles n'est pas cosmétique."))
    A(("ul", [
        "Le **prix brut** est le prix affiché à l'écran. Il chute mécaniquement le jour "
        "d'un split ou du détachement d'un dividende.",
        "Le **prix ajusté des splits** corrige les divisions d'actions.",
        "Le **prix total-return** corrige splits *et* dividendes : il représente la "
        "valeur d'un investissement qui réinvestit ses dividendes.",
    ]))
    A(("p", "**On a retenu le total-return.** Voici la démonstration, car c'est le genre "
           "d'argument qui distingue immédiatement un candidat sérieux."))
    A(("h3", "Pourquoi c'est décisif spécifiquement pour les utilities"))
    A(("p", "Les utilities versent 3 à 4 % de dividende par an, et surtout ce rendement "
           "est **très dispersé** d'un titre à l'autre. Notons δ<sub>A</sub> et "
           "δ<sub>B</sub> les rendements de dividende annuels de deux titres. Le prix "
           "brut décroche du prix total-return d'un facteur qui croît avec le temps :"))
    A(("math", "log P<sup>brut</sup>(t) ≈ log P<sup>TR</sup>(t) − δ · t"))
    A(("p", "Le spread calculé sur prix bruts contient donc un terme "
           "−(δ<sub>A</sub> − β δ<sub>B</sub>) · t : **une dérive linéaire déterministe**. "
           "Or une série qui dérive linéairement n'est pas stationnaire. Le test ADF la "
           "rejettera — et on écarterait ainsi des paires **authentiquement cointégrées**, "
           "uniquement parce que leurs politiques de dividende diffèrent."))
    A(("p", "Ordre de grandeur mesuré sur l'univers réel : le rendement de dividende "
           "implicite va de **1,07 % (PCG) à 6,39 % (OKE)**, soit 5,3 points d'écart entre "
           "deux titres du même secteur. Un écart de 2 points par an injecte déjà environ "
           "6 % de dérive sur une fenêtre de sélection de trois ans — du même ordre que "
           "l'amplitude du spread lui-même. L'effet n'est pas marginal."))
    A(("h3", "L'expérience, et ce qu'elle autorise réellement à conclure"))
    A(("p", "On a testé les 465 paires sur une fenêtre in-sample type (2016-2018), une fois "
           "sur prix total-return, une fois sur prix bruts, et comparé les **sélections** "
           "obtenues :"))
    A(("code",
       "465 paires testees sur 2016-2018 (Engle-Granger, seuil 5 %)\n"
       "\n"
       "  significatives sur TOTAL-RETURN : 112\n"
       "  significatives sur PRIX BRUTS   : 137\n"
       "  desaccord : 51 paires (11 %)  --  13 perdues, 38 creees\n"
       "\n"
       "  mais l'essentiel est un alea de p-value pres du seuil :\n"
       "     perdues : 11/13 marginales (85 %)\n"
       "     creees  : 26/38 marginales (68 %)\n"
       "\n"
       "  BASCULEMENTS FRANCS (p > 0.15 d'un cote, < 0.05 de l'autre)\n"
       "     paires FABRIQUEES par les prix bruts : 16\n"
       "     paires DETRUITES  par les prix bruts :  1"))
    A(("p", "**Il faut lire ce résultat avec précaution**, et c'est un point de méthode "
           "important. La conclusion tentante — « les prix bruts fabriquent 38 fausses "
           "paires » — n'est pas soutenable : 68 % de ces paires ont une p-value dans "
           "l'intervalle [0,03 ; 0,08]. Elles basculent parce qu'une p-value proche du "
           "seuil est bruitée, pas parce que la dérive de dividende aurait changé quoi que "
           "ce soit de substantiel. Les compter comme des erreurs de sélection serait "
           "malhonnête."))
    A(("p", "**Ce qui résiste, c'est l'asymétrie des basculements francs : 16 contre 1.** "
           "Seize paires sans relation d'équilibre sur les prix économiquement corrects "
           "deviennent franchement significatives dès qu'on utilise les prix bruts ; une "
           "seule bascule dans l'autre sens. La corrélation entre l'ampleur de la dérive et "
           "l'écart de p-value vaut 0,29 — le mécanisme est là, mais il n'explique qu'une "
           "partie du phénomène."))
    A(("p", "**Le mécanisme, pour les cas francs.** Si deux titres divergent économiquement "
           "mais que celui qui monte le plus verse aussi le plus de dividende, la dérive "
           "**annule** la divergence sur prix bruts. Le spread devient artificiellement "
           "plat, donc « stationnaire » au sens du test."))
    A(("key", "La formulation à retenir",
       ["Les deux séries de prix produisent des sélections qui diffèrent sur 11 % des "
        "paires. Dire « 38 fausses paires » serait exagéré — l'essentiel est du bruit de "
        "seuil. Mais l'asymétrie **16 contre 1** est robuste : les prix bruts fabriquent "
        "de la cointégration apparente bien plus souvent qu'ils n'en détruisent.",
        "Le choix de la série de prix n'est pas un détail de préparation des données : "
        "**c'est une décision qui change la sélection elle-même.**",
        "Savoir distinguer « ce que je voudrais conclure » de « ce que mes données "
        "autorisent » est précisément ce qu'un fonds cherche à tester en entretien."]))

    A(("h2", "6.2 Pourquoi on ne remplit jamais les prix manquants"))
    A(("p", "Quand une série a un trou — jour férié partiel, suspension de cotation, "
           "défaut du fournisseur — le réflexe universel est le *forward-fill* : reporter "
           "le dernier prix connu. On ne le fait pas ici. Mais la raison n'est pas celle "
           "qu'on lit partout, et cette section mérite d'être lue attentivement parce "
           "qu'elle contient une **erreur commise puis corrigée**."))
    A(("h3", "L'argument habituel — et pourquoi il est faux"))
    A(("p", "L'argument que l'on trouve dans la plupart des tutoriels, et que j'ai moi-même "
           "avancé avant de le tester, est le suivant : reporter un prix crée un rendement "
           "nul suivi d'un rattrapage, donc la séquence « pas de mouvement puis mouvement "
           "compensatoire », c'est-à-dire la signature du retour à la moyenne. Le "
           "forward-fill fabriquerait donc le signal que la stratégie cherche."))
    A(("p", "**C'est faux, et la simulation le montre sans ambiguïté.** On simule une marche "
           "aléatoire pure — donc sans aucun retour à la moyenne — on y introduit 5 % de "
           "jours manquants, on applique un forward-fill, et on teste :"))
    A(("code",
       "Marche aleatoire pure, 400 simulations, 5 % de jours manquants\n"
       "\n"
       "  taux de rejet ADF a 5 %   sans ffill :   6.8 %\n"
       "  taux de rejet ADF a 5 %   avec ffill :   6.5 %\n"
       "  autocorr. lag-1 des diffs sans ffill : -0.0016\n"
       "  autocorr. lag-1 des diffs avec ffill : -0.0017"))
    A(("p", "Aucun effet. Et sur un spread réellement mean-reverting (processus "
           "d'Ornstein-Uhlenbeck de demi-vie 15 jours), le forward-fill ne biaise ni la "
           "demi-vie estimée (+0,2 %) ni le nombre de signaux générés (67,6 contre 67,8)."))
    A(("p", "La raison est simple une fois qu'on la voit : le forward-fill **redistribue** "
           "les incréments sans les modifier. Le jour manquant porte un rendement nul, le "
           "jour suivant porte la somme des deux incréments. La somme totale est "
           "inchangée, et comme les incréments sont indépendants, la covariance entre "
           "rendements consécutifs reste nulle. Il n'y a pas de « rattrapage "
           "compensatoire » : il y a un report."))
    A(("warn", "Ce que cela veut dire pour l'entretien",
       ["Si tu avances l'argument du « retour à la moyenne fabriqué » face à quelqu'un qui "
        "a déjà fait la simulation, tu perds toute crédibilité sur le reste du projet.",
        "L'argument correct est plus simple et plus solide — et le fait de pouvoir dire "
        "« je l'ai cru, je l'ai testé, c'était faux, voici ce qui est vrai » vaut "
        "davantage que n'importe quelle justification apprise."]))
    A(("h3", "L'argument correct : la tradabilité"))
    A(("p", "Le vrai problème du forward-fill n'est pas statistique, il est **opérationnel**. "
           "Sur un spread d'Ornstein-Uhlenbeck avec 5 % de jours manquants :"))
    A(("code",
       "Spread OU (demi-vie 15 j), 5 % de jours manquants, 300 simulations\n"
       "\n"
       "  demi-vie estimee  sans ffill : 13.95 j\n"
       "  demi-vie estimee  avec ffill : 13.98 j     (biais +0.2 %)\n"
       "  signaux |z| > 2   sans ffill : 67.8\n"
       "  signaux |z| > 2   avec ffill : 67.6\n"
       "\n"
       "  signaux tombant un jour SANS COTATION : 2.6 par serie\n"
       "  soit 3.8 % des signaux, a un prix qui n'existe pas"))
    A(("key", "La justification à retenir",
       ["Le forward-fill ne corrompt pas la statistique du spread : il corrompt "
        "l'**exécution**. Environ 4 % des signaux d'entrée tombent un jour où le titre ne "
        "cotait pas. Le backtest ouvre alors une position à un prix qui n'a jamais existé, "
        "puis enregistre un P&L sur cette transaction impossible.",
        "C'est une fuite d'exécution, de la même famille que le fait de trader au cours du "
        "jour où le signal est calculé — le sujet de la Phase 6.",
        "**La décision : aucun forward-fill.** Les valeurs manquantes restent des NaN et "
        "sont traitées explicitement, paire par paire, au moment du test."]))

    A(("h2", "6.3 Le nettoyage, et pourquoi il est volontairement minimal"))
    A(("p", "Le nettoyage appliqué se réduit à deux règles, choisies pour être **non "
           "destructives** — un nettoyage agressif est lui-même une source de biais."))
    A(("ol", [
        "**Suppression des colonnes entièrement vides.** Ce sont les tickers pour "
        "lesquels le fournisseur ne sert plus rien, typiquement des sociétés délistées. "
        "On ne les efface pas de la mémoire du projet : ils sont comptabilisés dans le "
        "rapport de survivorship (section 9).",
        "**Suppression des dates où moins de 80 % des titres *en vie* cotent.** Ce sont "
        "des jours fériés partiels ou des incidents de données, pas des séances. "
        "Le point subtil est le dénominateur : on compare au nombre de titres **en vie à "
        "cette date**, et non au nombre total de colonnes. Sinon, un titre introduit en "
        "2022 ferait échouer le critère sur toutes les séances de 2014 à 2021 et on "
        "supprimerait huit ans de données valides.",
    ]))

    # ====================================================================== #
    A(("h1", "7. Décision n°4 — L'univers point-in-time", True))
    A(("h2", "7.1 Le problème : pourquoi le survivorship est une fuite"))
    A(("p", "Prendre les constituants **actuels** d'un indice et remonter leur historique "
           "est la méthode employée dans l'immense majorité des projets étudiants. Elle "
           "est fausse, et la raison est plus profonde qu'un simple « biais »."))
    A(("p", "Sélectionner aujourd'hui les sociétés membres du S&P 500 revient à utiliser "
           "une information datée d'aujourd'hui — **le fait que ces sociétés existent "
           "encore et sont assez grandes pour figurer dans l'indice** — afin de décider "
           "quoi trader en 2015. C'est structurellement identique à un look-ahead sur le "
           "signal. La seule différence est que la variable qui fuit n'est pas un prix "
           "futur : c'est la **survie**."))
    A(("p", "L'effet est systématique et va toujours dans le même sens : on exclut "
           "précisément les sociétés qui ont fait faillite, ont été rachetées ou ont "
           "décroché — c'est-à-dire les cas où une relation de cointégration se brise "
           "violemment. On mesure donc la performance de la stratégie sur un échantillon "
           "amputé de ses pires scénarios."))

    A(("h2", "7.2 Deux méthodes possibles, et pourquoi on a changé"))
    A(("p", "**Méthode A — reconstruire par les changements.** On part de la liste actuelle "
           "et on remonte le temps en inversant chaque ajout et chaque retrait consignés "
           "dans une table historique. C'est ce qui était prévu initialement."))
    A(("p", "Cette table n'existant plus sur la source publique, on a basculé sur une "
           "**méthode B, qui s'avère strictement supérieure** : lire l'historique des "
           "révisions de la page. La page listant les constituants du S&P 500 est éditée "
           "en continu depuis plus de quinze ans, et chaque version passée reste "
           "accessible. On récupère donc la page **telle qu'elle existait** à chaque date "
           "de sélection."))
    A(("table",
       ["", "Méthode A — table des changements", "Méthode B — révisions (retenue)"],
       [["Principe", "Inverser les changements depuis aujourd'hui",
         "Lire la page telle qu'elle était à la date voulue"],
        ["Secteur GICS", "Perdu pour les sociétés sorties : la table ne consigne que le "
         "ticker, il faudrait réattribuer le secteur à la main",
         "**Observé directement**, tel qu'il était classé à l'époque"],
        ["Reclassifications", "Invisibles : un titre reclassé d'un secteur à l'autre "
         "n'apparaît pas comme un changement d'indice",
         "**Capturées automatiquement** (cas ONEOK, section 8.4)"],
        ["Robustesse", "Une erreur dans la table se propage à toutes les dates antérieures",
         "Chaque date est lue indépendamment : une erreur reste locale"]],
       [0.16, 0.42, 0.42],
       "La bascule vers la méthode B n'était pas un pis-aller : elle donne l'appartenance "
       "*et* le secteur point-in-time, tous deux directement observés."))

    A(("h2", "7.3 Le détail technique qui garantit l'absence de fuite"))
    A(("p", "Toute la propriété anti-fuite tient dans un paramètre de la requête :"))
    A(("code",
       'params = {\n'
       '    "rvstart": timestamp,   # la date de selection\n'
       '    "rvdir":   "older",     # <-- REMONTER dans le temps depuis cette date\n'
       '    "rvlimit": 1,\n'
       '}'))
    A(("p", "`rvdir=\"older\"` demande la dernière révision publiée **avant** la date de "
           "sélection. Avec `\"newer\"`, on obtiendrait la première révision publiée "
           "*après* — donc une page pouvant déjà refléter des changements d'indice "
           "postérieurs à la date de décision. Un seul mot sépare une reconstruction "
           "correcte d'une fuite."))
    A(("p", "Le décalage observé entre la date de sélection et la révision utilisée est de "
           "**2 jours en médiane, 19 jours au maximum**. Ce décalage joue toujours dans le "
           "sens conservateur : la page utilisée est légèrement *en retard* sur la "
           "réalité, jamais en avance."))

    A(("h2", "7.4 Trois sous-décisions de mise en œuvre"))
    A(("h3", "Cadence semestrielle"))
    A(("p", "On prend une photo de l'indice tous les six mois. La re-sélection du "
           "walk-forward étant annuelle, chaque date de sélection dispose d'une photo "
           "vieille de six mois au maximum. Descendre au mensuel multiplierait par six le "
           "nombre de requêtes pour une précision dont la stratégie n'a aucun usage : elle "
           "ne rebalance qu'une fois par an."))
    A(("h3", "Cache disque systématique"))
    A(("p", "Chaque révision téléchargée est écrite sur disque. Ce n'est pas une "
           "optimisation, c'est une **exigence de reproductibilité** : sans cache, la "
           "construction de l'univers dépend d'une page web modifiable à tout instant, et "
           "peut même être éditée rétroactivement. Deux exécutions à un mois d'intervalle "
           "pourraient produire deux univers différents, donc deux backtests différents, "
           "sans qu'aucune ligne de code n'ait changé."))
    A(("h3", "Échec bruyant en cas de requête ratée"))
    A(("p", "Si une requête échoue de façon répétée, le programme lève une exception et "
           "s'arrête. La tentation serait de passer le snapshot manquant et de continuer. "
           "Ce serait précisément le type de dégradation silencieuse contre laquelle toute "
           "la Phase 1 est construite : un univers tronqué sur une période, sans que rien "
           "ne le signale."))

    # ====================================================================== #
    A(("h1", "8. Décision n°5 — Le contrôle d'identité des tickers", True))
    A(("h2", "8.1 Ce qu'on a découvert"))
    A(("p", "Une fois l'univers point-in-time construit, on a téléchargé les prix des "
           f"{n_union} tickers. L'examen des dates de début a révélé un problème qui "
           "n'était pas anticipé et qui aurait invalidé l'intégralité du projet."))
    A(("table",
       ["Ticker", "Société membre de l'indice", "Sortie", "Données servies à partir de", "Ce que c'est en réalité"],
       [["`NU`", "Northeast Utilities", "2015", "9 décembre 2021",
         "**Nu Holdings** — néobanque brésilienne, IPO fin 2021"],
        ["`POM`", "Pepco Holdings", "2016 (racheté par Exelon)", "8 octobre 2025",
         "Une société sans rapport, cotée depuis 2025"],
        ["`TE`", "TECO Energy", "2016 (racheté par Emera)", "10 janvier 2020",
         "Une société sans rapport"],
        ["`TEG`", "Integrys Energy", "2015 (racheté par WEC)", "22 décembre 2015",
         "Une société sans rapport, ticker repris quelques mois après"]],
       [0.09, 0.24, 0.20, 0.20, 0.27],
       "Les quatre cas de recyclage de ticker détectés dans l'univers."))
    A(("p", "Les tickers ne sont pas des identifiants pérennes. Quand une société "
           "disparaît, son symbole retourne au pot commun et peut être réattribué. Trois "
           "lettres n'identifient pas une entreprise — elles identifient un emplacement "
           "sur un marché à un instant donné."))
    A(("warn", "Ce qui se serait passé sans ce contrôle",
       ["Le backtest aurait traité les prix d'une **néobanque brésilienne** comme ceux "
        "d'une utility américaine régulée. Il aurait testé la cointégration entre Nu "
        "Holdings et Duke Energy, l'aurait peut-être trouvée significative sur une "
        "fenêtre, et aurait ouvert des positions.",
        "Aucune exception n'aurait été levée. Aucun test n'aurait échoué. La courbe de "
        "P&L aurait été parfaitement lisse. Et en entretien, on aurait défendu avec "
        "conviction un résultat entièrement faux.",
        "C'est l'illustration exacte de la thèse de la section 3 : **les erreurs qui "
        "comptent ne provoquent pas d'erreur.**"]))

    A(("h2", "8.2 La règle de détection"))
    A(("p", "L'idée : on dispose de deux informations indépendantes sur chaque ticker. "
           "D'un côté, sa **fenêtre d'appartenance** au secteur, issue de la "
           "reconstruction point-in-time. De l'autre, sa **période de cotation**, issue du "
           "fournisseur de prix. Si le ticker désigne bien la même société dans les deux "
           "sources, ces deux intervalles doivent se recouvrir."))
    A(("math", "recouvrement = |[début cotation, fin cotation] ∩ [entrée indice, sortie indice]| "
               "/ |[entrée indice, sortie indice]|"))
    A(("ul", [
        "recouvrement **quasi nul** → le ticker désigne une autre société : `RECYCLED`",
        "recouvrement **partiel** (< 50 %) → historique tronqué, revue manuelle : `SUSPECT`",
        "recouvrement **substantiel** → identité cohérente : `OK`",
    ]))
    A(("p", "Le contrôle est **bloquant** : les tickers classés `RECYCLED`, `SUSPECT` ou "
           "`NO_DATA` sont retirés de l'univers tradable, et la liste des exclusions est "
           "écrite dans un rapport auditable."))

    A(("h2", "8.3 Les deux bugs rencontrés en écrivant ce contrôle"))
    A(("p", "Ils méritent d'être racontés, parce qu'ils illustrent que même un contrôle "
           "anti-erreur doit lui-même être vérifié."))
    A(("h3", "Bug n°1 — le faux positif ONEOK"))
    A(("p", "La première version classait `OKE` comme recyclé. Or ONEOK est bien la même "
           "société, avec un historique complet. L'erreur venait de la définition de la "
           "fenêtre d'appartenance : `[premier snapshot membre, dernier snapshot membre]`. "
           "ONEOK n'apparaissant que dans **un seul** snapshot — il a été reclassé de "
           "*Utilities* vers *Energy* par GICS début 2014 — sa fenêtre avait une durée "
           "nulle, et le recouvrement divisait par zéro."))
    A(("p", "Correction : être membre à la date *t* signifie l'être au moins jusqu'à la "
           "photo suivante. La fenêtre est donc élargie de six mois à droite. Le cas ONEOK "
           "est d'ailleurs intéressant en soi : c'est une **reclassification sectorielle**, "
           "que la méthode B capture automatiquement alors que la méthode A l'aurait "
           "totalement manquée."))
    A(("h3", "Bug n°2 — le seuil à zéro exact"))
    A(("p", "La règle initiale déclarait recyclé un recouvrement **exactement nul**. "
           "`TEG` a un recouvrement de 0,014 : Integrys a été racheté en juin 2015 et le "
           "ticker réattribué dès décembre 2015, si bien que les deux intervalles se "
           "chevauchent de dix jours. Avec un seuil à zéro, TEG passait entre les mailles."))
    A(("p", "Correction : seuil à 5 %. La leçon générale est qu'un test binaire fondé sur "
           "une égalité exacte est fragile — il faut une marge, et il faut avoir regardé "
           "les données pour la calibrer."))

    A(("h2", "8.4 L'angle mort qui subsiste"))
    A(("p", "Le contrôle compare deux intervalles. Il détecterait donc mal un recyclage "
           "survenant **pendant** la fenêtre d'appartenance — le recouvrement resterait "
           "élevé et le statut passerait à `OK`. Ce cas n'est pas observé ici, et il est "
           "peu probable en pratique (un ticker n'est pas réattribué instantanément), mais "
           "il est documenté plutôt que passé sous silence."))
    A(("p", "Le contrôle réellement robuste utiliserait un identifiant pérenne — CUSIP, "
           "SEDOL, ou le PERMNO du CRSP — qui ne change pas quand le ticker change. C'est "
           "exactement ce que fournissent les bases professionnelles, et c'est une des "
           "raisons pour lesquelles elles coûtent cher."))

    # ====================================================================== #
    A(("h1", "9. Les résultats de la Phase 1", True))
    A(("h2", "9.1 Les chiffres"))
    A(("table",
       ["Grandeur", "Valeur", "Ce qu'elle sert"],
       [["Tickers en union brute", f"{n_union}", "Tous les noms ayant appartenu au secteur sur la période"],
        ["Tickers tradables", f"**{n_trad}**", "Après contrôle d'identité"],
        ["Taille du secteur par date", f"{n_min} à {n_max}", "Varie avec les entrées/sorties d'indice"],
        ["Paires testables au maximum", f"**{pairs_max}**", "N(N−1)/2 — la charge de tests multiples de la Phase 3"],
        ["Faux positifs attendus à α = 5 %", f"**≈ {fp_max:.0f}**", "Le chiffre d'ouverture de la Phase 3"],
        ["Séances de bourse", f"{len(close)}", f"{close.index[0]:%d/%m/%Y} → {close.index[-1]:%d/%m/%Y}"],
        ["Photos de l'indice", f"{len(memb)}", "Semestrielles, lag médian 2 jours"]],
       [0.36, 0.16, 0.48]))
    A(("key", f"Le chiffre à retenir pour la Phase 3",
       [f"Avec {n_max} titres, on teste {pairs_max} paires. Sous l'hypothèse nulle "
        f"— aucune paire n'est réellement cointégrée — un test au seuil de 5 % en "
        f"déclarerait tout de même **≈ {fp_max:.0f} significatives**, par pur hasard.",
        "Tout l'argument de la Phase 3 consiste à comparer ce nombre à celui des paires "
        "effectivement trouvées significatives. Si on en trouve 30, on n'a presque rien "
        "trouvé."]))

    A(("h2", "9.2 Le cas PG&E : ce que le point-in-time achète concrètement"))
    A(("p", "La reconstruction montre que `PCG` est **hors de l'indice de juillet 2019 à "
           "juillet 2022** : PG&E a déposé le bilan en janvier 2019 après les incendies "
           "de Californie, a été retiré du S&P 500, puis réintégré après sa sortie de "
           "faillite."))
    A(("p", "PCG étant membre **aujourd'hui**, un univers bâti sur les constituants "
           "actuels l'inclurait sur toute la période, et le traderait donc **pendant sa "
           "faillite** — en sachant, implicitement, qu'il en est ressorti vivant et que "
           "son cours a fini par se redresser. C'est du look-ahead à l'état pur, et il "
           "porte sur l'un des épisodes les plus violents du secteur sur la période."))
    A(("p", "La reconstruction point-in-time l'exclut automatiquement, sans qu'aucune "
           "règle spécifique n'ait été écrite pour ce cas. C'est le signe qu'on a corrigé "
           "la cause et non le symptôme."))

    A(("h2", "9.3 Le survivorship résiduel, quantifié et orienté"))
    A(("p", f"Sur {n_union} tickers en union, **{len(lost)} sont exclus** : "
           f"{', '.join('`'+t+'`' for t in lost)}."))
    A(("table",
       ["Ticker", "Statut", "Ce qui s'est réellement passé"],
       [["`GAS`", "Aucune donnée", "AGL Resources, racheté par Southern Company (2016). Le fournisseur ne sert plus rien : perte sèche."],
        ["`NU`", "Recyclé", "Northeast Utilities, **renommé Eversource (ES)** en 2015. Perte apparente seulement : la série `ES` remonte à 2014 et porte l'historique complet."],
        ["`POM`", "Recyclé", "Pepco Holdings, racheté par Exelon (2016)."],
        ["`TE`", "Recyclé", "TECO Energy, racheté par Emera (2016)."],
        ["`TEG`", "Recyclé", "Integrys Energy, racheté par WEC (2015)."]],
       [0.10, 0.14, 0.76]))
    A(("p", f"En volume : {slots_all} paires (photo × ticker) dans l'univers brut, "
           f"{slots_kept} conservées, soit **{100*(slots_all-slots_kept)/slots_all:.1f} % "
           "de pertes**. Ces pertes sont **entièrement concentrées sur 2014-2016** et "
           "**nulles à partir de janvier 2017**. Au pire snapshot, 5 noms manquent sur 31, "
           "soit environ 30 % des paires potentielles non testables."))
    A(("h3", "La direction du biais — le point le plus important de la section"))
    A(("p", "Les cinq noms perdus sont tous sortis de l'indice **par acquisition**, "
           "aucun par faillite. Or une OPA est précisément l'événement qui brise une "
           "relation de cointégration de la manière la plus brutale : le cours de la cible "
           "saute au prix d'offre, puis se fige, définitivement décorrélé de son secteur. "
           "Une position vendeuse sur la cible au moment de l'annonce subit une perte "
           "instantanée qui ne se résorbe **jamais** — le retour à la moyenne n'a pas lieu."))
    A(("key", "La formulation à retenir",
       ["Exclure ces cinq noms retire de l'échantillon des scénarios de perte extrême et "
        "irréversible. **Le biais résiduel surestime donc la performance de la stratégie.**",
        "C'est le cas le plus dangereux : un biais favorable ne se signale jamais de "
        "lui-même. On n'a aucune raison de le chercher, puisque les résultats sont bons.",
        "Conséquence pratique : le premier fold (sélection 2014-2016) est le plus exposé. "
        "L'honnêteté consiste à reporter son résultat **séparément** plutôt que de "
        "l'agréger silencieusement aux autres."]))

    A(("h2", "9.4 Deux zones grises assumées"))
    A(("ul", [
        "**EVRG.** Evergy est né en 2018 de la fusion de Westar et Great Plains. Le "
        "fournisseur sert un historique remontant à 2014 : ce sont les prix de Westar, le "
        "prédécesseur. C'est défendable — le prédécesseur est l'objet économique "
        "pertinent — mais il faut le savoir si un interviewer creuse.",
        "**SCG.** SCANA affiche un recouvrement de 0,69, inférieur à 1 : l'historique "
        "servi commence en juillet 2015 alors que la société était membre depuis 2014. "
        "L'historique est tronqué par le fournisseur, mais l'identité est cohérente. Le "
        "titre est conservé, avec moins de données que les autres.",
    ]))

    # ====================================================================== #
    A(("h1", "10. Ce qui a été délibérément reporté", True))
    A(("h2", "10.1 Le filtre de liquidité"))
    A(("p", "Le plan prévoit un univers d'actions **liquides**. Aucun filtre de liquidité "
           "n'a pourtant été appliqué en Phase 1. C'est délibéré, et la raison est une "
           "fuite."))
    A(("p", "Filtrer maintenant reviendrait à calculer le volume médian sur **toute** la "
           "période 2014-2026, puis à ne garder que les titres au-dessus d'un seuil. On "
           "utiliserait donc, pour décider quoi trader en 2015, l'information « ce titre "
           "sera encore liquide en 2025 ». Cette information est fortement corrélée à la "
           "survie et au succès de l'entreprise : c'est un survivorship bias déguisé en "
           "critère technique."))
    A(("p", "**Le filtre doit être recalculé à chaque fold, sur la fenêtre in-sample "
           "uniquement**, exactement comme la sélection des paires. Il arrivera donc en "
           "Phase 3, au moment où la sélection elle-même est construite. C'est pour cette "
           "raison que la Phase 1 s'arrête sur un univers non filtré — ce n'est pas un "
           "oubli."))
    A(("h2", "10.2 Récapitulatif des garde-fous en place"))
    A(("table",
       ["Garde-fou", "Contre quoi", "Où"],
       [["Appartenance point-in-time", "Survivorship sur la composition", "`src/universe.py`"],
        ["Contrôle d'identité bloquant", "Recyclage de tickers", "`validate_identity`"],
        ["Prix total-return", "Dérive de dividende dans le spread", "`src/data.py`"],
        ["Aucun forward-fill", "Signaux générés les jours sans cotation", "`src/data.py`"],
        ["Dénominateur « titres en vie »", "Suppression de séances valides", "`_clean`"],
        ["Cache disque", "Non-reproductibilité de l'univers", "`data/raw/wiki/`"],
        ["Échec bruyant sur requête ratée", "Univers silencieusement tronqué", "`_get`"],
        ["Décision d'univers datée", "Data snooping au niveau du secteur", "`docs/01_universe_decision.md`"],
        ["Filtre de liquidité reporté", "Survivorship déguisé en critère technique", "Phase 3"]],
       [0.30, 0.42, 0.28]))

    # ====================================================================== #
    A(("h1", "11. Les deux questions posées — réponses détaillées", True))
    A(("h2", "Question 1 — L'angle mort du contrôle d'identité"))
    A(("p", "*« Quel cas de recyclage passerait à travers la règle ? Et par quel mécanisme "
           "concret le P&L deviendrait-il faux sans qu'aucun test ne le détecte ? »*"))
    A(("h3", "Le cas qui passe"))
    A(("p", "Un recyclage survenant **pendant** la fenêtre d'appartenance. La règle mesure "
           "le recouvrement entre la période de cotation et la fenêtre d'appartenance ; si "
           "la société X est membre de 2014 à 2020 et disparaît en 2017, un ticker "
           "réattribué en 2018 à une société Y produirait une série continue de 2014 à "
           "2020, recouvrant parfaitement la fenêtre. Statut : `OK`."))
    A(("h3", "Le mécanisme de contamination du P&L"))
    A(("p", "La série concaténée présenterait une **discontinuité de niveau** à la date de "
           "bascule : le prix passe de celui de X à celui de Y, sans lien économique. "
           "Trois conséquences en cascade :"))
    A(("ol", [
        "Le test de cointégration sur la fenêtre in-sample verrait un spread avec un saut. "
        "Selon la position du saut, le test **rejetterait** à tort (le saut ressemble à "
        "une racine unitaire) ou **accepterait** à tort si le saut est suivi d'un régime "
        "stable qui ressemble à un retour à la moyenne.",
        "Si la paire est sélectionnée, le z-score au moment du saut atteindrait une valeur "
        "extrême — typiquement |z| > 5. La stratégie ouvrirait une position **maximale** "
        "sur ce qu'elle interprète comme une opportunité exceptionnelle.",
        "Cette position ne se déboucle jamais, puisque le saut n'est pas un écart "
        "temporaire mais un changement d'objet. Le P&L enregistre un gain ou une perte "
        "massive, purement fictive.",
    ]))
    A(("p", "**Pourquoi aucun test ne le détecte :** tous les diagnostics de la Phase 7 "
           "(Sharpe, drawdown, hit rate, turnover) sont des statistiques *agrégées* "
           "calculées **sur cette même série corrompue**. Ils sont cohérents entre eux et "
           "avec les données. Rien n'est incohérent — les données sont simplement fausses. "
           "Aucune quantité de contrôles internes ne peut détecter une erreur qui affecte "
           "l'entrée du système."))
    A(("p", "**Ce qui le détecterait :** un identifiant pérenne (CUSIP, SEDOL, PERMNO), "
           "ou à défaut un contrôle de **discontinuité de rendement** — un rendement "
           "quotidien de plusieurs centaines de pourcents non associé à un split connu est "
           "la signature d'un changement d'entité. C'est un contrôle bon marché qu'il "
           "serait raisonnable d'ajouter."))

    A(("h2", "Question 2 — L'objection sur PG&E"))
    A(("p", "*« Un vrai fonds n'est pas contraint par l'appartenance au S&P 500 ; il aurait "
           "pu trader PG&E pendant sa faillite. En l'excluant, tu n'as pas corrigé un "
           "biais, tu en as introduit un autre. »*"))
    A(("p", "L'objection est juste sur les faits et fausse sur la conclusion. Il faut "
           "accorder le premier point avant de répondre."))
    A(("h3", "Ce que l'appartenance à l'indice sert réellement à représenter"))
    A(("p", "On n'utilise pas l'appartenance au S&P 500 parce qu'un mandat l'imposerait. On "
           "l'utilise comme **variable de substitution, observable et point-in-time**, "
           "pour un ensemble de propriétés qu'on veut réellement imposer et qui sont "
           "difficiles à mesurer directement : capitalisation suffisante, flottant "
           "suffisant, liquidité permettant d'entrer et de sortir sans impact, et "
           "possibilité effective de **vendre à découvert** — c'est-à-dire l'existence "
           "d'un stock de titres empruntables à un coût raisonnable."))
    A(("p", "Le point décisif est ce dernier. Le pairs trading exige une jambe courte. "
           "Pendant une procédure de faillite, le titre devient extrêmement difficile et "
           "coûteux à emprunter : le coût de l'emprunt peut atteindre des dizaines de "
           "pourcents annualisés, et le prêteur peut rappeler ses titres à tout moment, "
           "forçant un rachat au pire moment (*short squeeze*). Un backtest qui ignore ces "
           "coûts et les tarifie à zéro **surestime** massivement la performance."))
    A(("h3", "La réponse en trois temps"))
    A(("ol", [
        "**Accorder.** Oui, l'appartenance à l'indice n'est pas la contrainte réelle d'un "
        "fonds. C'est une approximation.",
        "**Retourner l'argument.** C'est une approximation **conservatrice**. Elle exclut "
        "les périodes où l'exécution — en particulier l'emprunt de titres — est la plus "
        "coûteuse et la plus incertaine. En incluant PG&E pendant sa faillite avec des "
        "coûts de financement supposés normaux, je ne serais pas plus réaliste : je serais "
        "**plus optimiste**, en tarifant à zéro le risque le plus cher du marché.",
        "**Nommer le vrai correctif.** La solution rigoureuse n'est pas d'inclure PG&E "
        "sans précaution, mais de modéliser explicitement le coût d'emprunt et la "
        "disponibilité du titre. Cela exige des données de *securities lending*, qui sont "
        "payantes. En leur absence, exclure est le choix conservateur, et je le documente.",
    ]))
    A(("key", "Le principe général derrière cette réponse",
       ["Face à une objection méthodologique fondée, la mauvaise réponse est de nier ; la "
        "réponse médiocre est de concéder ; **la bonne réponse est de montrer dans quel "
        "sens le choix biaise le résultat, et de préférer systématiquement le sens "
        "défavorable à la stratégie.**",
        "C'est la ligne directrice de toute la Phase 1, et c'est le message que le projet "
        "doit transmettre en entier."]))

    # ====================================================================== #
    A(("h1", "12. Glossaire", True))
    A(("table",
       ["Terme", "Définition"],
       [["Spread", "Écart entre les log-prix de deux titres, pondéré par le ratio de couverture : log P_A − β log P_B."],
        ["Ratio de couverture (β)", "Poids de la seconde jambe. Répond à : combien vendre de B pour un dollar de A."],
        ["I(0) / I(1)", "Série stationnaire / série dont seule la différence première est stationnaire. Un prix d'action est typiquement I(1)."],
        ["Stationnarité", "Propriété d'une série dont la loi ne dépend pas du temps : elle oscille autour d'une moyenne fixe au lieu de dériver."],
        ["Cointégration", "Existence d'une combinaison linéaire I(0) de séries I(1). Formalise l'idée d'une relation d'équilibre de long terme."],
        ["ADF", "Test de Dickey-Fuller augmenté. Teste la présence d'une racine unitaire ; on l'applique au résidu pour tester sa stationnarité."],
        ["Look-ahead", "Utilisation, pour décider à la date t, d'une information non disponible à t. Gonfle la performance sans se signaler."],
        ["Survivorship bias", "Forme de look-ahead portant sur l'existence : ne retenir que les sociétés ayant survécu jusqu'à aujourd'hui."],
        ["Point-in-time", "Donnée reconstruite telle qu'elle était observable à la date considérée, et non telle qu'on la connaît aujourd'hui."],
        ["Walk-forward", "Protocole de validation où sélection et évaluation se succèdent dans le temps, sans jamais se chevaucher."],
        ["In-sample / Out-of-sample", "Période de sélection et d'estimation / période de test, postérieure et strictement disjointe."],
        ["Total-return", "Série de prix ajustée des splits et des dividendes réinvestis."],
        ["Forward-fill", "Remplissage d'une valeur manquante par la dernière valeur connue. Proscrit ici (section 6.2)."],
        ["Merchant (producteur)", "Producteur d'électricité vendant au prix de marché, par opposition à un régulé dont les tarifs sont fixés."],
        ["GICS", "Nomenclature sectorielle standard. Utilities est le secteur de niveau 1 retenu ici."],
        ["Recyclage de ticker", "Réattribution d'un symbole boursier à une autre société après la disparition de la première."]],
       [0.22, 0.78]))
    return B
