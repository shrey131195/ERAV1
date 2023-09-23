# Transformer Model for English-French Translation

This repository contains code for training a Transformer model for English to French translation. The goal is to create a model that achieves a loss under 1.8 on a specific dataset.

## Dataset
The dataset used for this project is the "en-fr" dataset from Opus Books.

## Data Preprocessing
Before training the Transformer model, the following preprocessing steps were performed on the dataset:

1. **Removing Long English Sentences**: All English sentences with more than 150 tokens were removed from the dataset.

2. **Removing Corresponding French Sentences**: For each English sentence removed, the corresponding French sentence was also removed if its length exceeded the length of the English sentence plus 10 tokens.

## Transformer Model
The core of this project is the implementation of a Transformer model for English-French translation. The model includes an encoder-decoder architecture with parameter sharing (PS) and utilizes mixed-precision training with the Automatic Mixed Precision (AMP) library.

## Training Logs

Below is a summary of training progress for the last 5 epochs, including metrics such as loss, accuracy, or any other relevant information. This can help you monitor the model's performance during training.

Epoch 0: train_loss=5.0322
Validation: 0it [00:00, ?it/s]
*****************************************
    SOURCE: "We are all mortal," affirmed Grivet.
    TARGET: —Nous sommes tous mortels, affirma Grivet.
 PREDICTED: — Nous sommes tous tous , pensa Grivet .
*****************************************

*****************************************
    SOURCE: His companion also joked him about the girls, declaring that he had seen him with a putter in the wheat on the Bas-de-Soie side.
    TARGET: Son compagnon le plaisantait aussi sur les filles, jurait l'avoir vu avec une herscheuse dans les blés, du côté des Bas-de-Soie.
 PREDICTED: Son compagnon le , lui , lui , il l ' avait vu avec une herscheuse dans les tables de l ' .
*****************************************

Epoch 1: train_loss=4.2107
Validation: 0it [00:00, ?it/s]
*****************************************
    SOURCE: I cocked an ear and listened.
    TARGET: J'écoutais, je tendais l'oreille.
 PREDICTED: Je un oreille et l ' écoutais .
*****************************************

*****************************************
    SOURCE: Emma at first felt a great astonishment; then was anxious to be delivered that she might know what it was to be a mother.
    TARGET: Emma d’abord sentit un grand étonnement, puis eut envie d’être délivrée, pour savoir quelle chose c’était que d’être mère.
 PREDICTED: Emma , d ’ abord , sentit un grand étonnement ; puis elle était bien disposée à être payé qu ’ elle pouvait savoir ce qu ’ il était pour une mère .
*****************************************

Epoch 2: train_loss=4.2380
Validation: 0it [00:00, ?it/s]
*****************************************
    SOURCE: That is his mark.
    TARGET: C’est l’une de ses traces.
 PREDICTED: C ’ est sa montre .
*****************************************

*****************************************
    SOURCE: In the afternoon Lydia was urgent with the rest of the girls to walk to Meryton, and to see how everybody went on; but Elizabeth steadily opposed the scheme.
    TARGET: L’apres-midi, elle pressa vivement ses sours ainsi que les autres jeunes filles de venir faire un tour a Meryton, mais Elizabeth s’y opposa avec fermeté.
 PREDICTED: Dans le midi , Lydia était avec le reste des filles a Meryton pour voir comment le monde s ’ a Elizabeth .
*****************************************

Epoch 3: train_loss=3.7388
Validation: 0it [00:00, ?it/s]
*****************************************
    SOURCE: She broke forth as never moon yet burst from cloud: a hand first penetrated the sable folds and waved them away; then, not a moon, but a white human form shone in the azure, inclining a glorious brow earthward.
    TARGET: J'aperçus d'abord une main qui sortait des noirs plis du ciel et qui écartait les nuées; puis je vis, au lieu de la lune, une ombre blanche se dessinant sur un fond d'azur, et inclinant son noble front vers la terre.
 PREDICTED: Elle éclata comme la lune ne s ' éclata pas , une première main , les plis et les ; puis , une lune , une mare blanche , mais une forme d ' huile brillait dans l ' eau .
*****************************************

*****************************************
    SOURCE: I perceived at once that he was examining to find the exact place where the torrent could be heard the loudest.
    TARGET: Je compris qu'il cherchait le point précis où le torrent se faisait entendre plus bruyamment.
 PREDICTED: Je m ' apperçus qu ' il examinait à la place où le torrent était exacte .
*****************************************

Epoch 4: train_loss=3.3476
Validation: 0it [00:00, ?it/s]
*****************************************
    SOURCE: About the middle of the next day, as she was in her room getting ready for a walk, a sudden noise below seemed to speak the whole house in confusion; and, after listening a moment, she heard somebody running up stairs in a violent hurry, and calling loudly after her.
    TARGET: Dans l’apres-midi du jour suivant, pendant qu’elle s’habillait pour une promenade, un bruit soudain parut mettre toute la maison en rumeur ; elle entendit quelqu’un monter précipitamment l’escalier en l’appelant a grands cris.
 PREDICTED: Vers le milieu du jour suivant , comme elle était dans sa chambre , un bruit subit semblait parler toute la maison ; et , apres un moment , elle entendit un moment , elle entendit un bruit violent , et , après un bruit de pas , et qui , après tout à fait , et qui , après elle .
*****************************************

*****************************************
    SOURCE: Il pouvait à peine prononcer les mots de sa réponse.
    TARGET: It was all he could do to pronounce the words of his answer.
 PREDICTED: He might barely the words of his answer .
*****************************************

Epoch 5: train_loss=3.0245
Validation: 0it [00:00, ?it/s]
*****************************************
    SOURCE: "Do I inspire you with horror?" he repeated.
    TARGET: « Est-ce que je vous fais horreur ? » répéta-t-il.
 PREDICTED: -- Je vous aime d ' horreur ? répéta - t - il .
*****************************************

*****************************************
    SOURCE: "Come," said the king, "will you swear, by my father, that Athos was at your residence during the event and that he took no part in it?"
    TARGET: -- Voyons, dit le roi, me jurez-vous, par mon père, que M. Athos était chez vous pendant l'événement, et qu'il n'y a point pris part?
 PREDICTED: « Allons , dit le roi , vous le jurez , mon père , que Athos était chez vous pendant le cas où il ne s ' en est pas , n ' est - ce pas ?
*****************************************

Epoch 6: train_loss=2.4728
Validation: 0it [00:00, ?it/s]
*****************************************
    SOURCE: Je m’imaginais que Sherlock Holmes ne perdrait pas une minute pour se précipiter dans la maison et attaquer de front l’énigme qu’elle renfermait ; mais a mon grand étonnement, tout différente fut sa maniere de procéder.
    TARGET: With an air of nonchalance which, under the circumstances, seemed to me to border upon affectation, he lounged up and down the pavement, and gazed vacantly at the ground, the sky, the opposite houses and the line of railings.
 PREDICTED: I went to that Sherlock Holmes would not have a minute to be in the house and herself from the of the .
*****************************************

*****************************************
    SOURCE: The examination was easy, and Ayrton soon let the glass fall again, saying--
    TARGET: La vérification fut facile, et Ayrton laissa bientôt retomber sa lunette en disant:
 PREDICTED: L ' examen était facile , et Ayrton laissa retomber la lunette , en disant :
*****************************************

Epoch 7: train_loss=2.7195
Validation: 0it [00:00, ?it/s]
*****************************************
    SOURCE: Monsieur the bailiff is our friend.
    TARGET: Monsieur le bailli est notre ami.
 PREDICTED: Monsieur le bailli est notre ami .
*****************************************

*****************************************
    SOURCE: Sir:
    TARGET: « _Monsieur,_
 PREDICTED: Sir William .
*****************************************

Epoch 8: train_loss=2.3402
Validation: 0it [00:00, ?it/s]
*****************************************
    SOURCE: This evening, 'I am a galley slave,' he said to himself, as he entered it, with a vivacity long unfamiliar to him: 'let us hope that the second letter will be as boring as the first.'
    TARGET: Ce jour-là, j’ai un travail forcé, se dit-il en rentrant et avec une vivacité que depuis longtemps il ne connaissait plus : espérons que la seconde lettre sera aussi ennuyeuse que la première.
 PREDICTED: Ce soir , je suis un esclave , se dit - il en l ’ abordant avec une longue verve pour lui : Que notre principal n ’ en sera pas ainsi de peine que la première lettre ne sera pas aussi ennuyeux que la première .
*****************************************

*****************************************
    SOURCE: As we lay in the hollow two horsemen came spurring along the ridge right in front of us, riding as hard as hoof could rattle.
    TARGET: Comme nous étions couchés dans le fossé, deux cavaliers arrivèrent à fond de train, sur la crête, en face de nous.
 PREDICTED: Nous étions en train de , deux s ' sur le droite devant nous , de grands pas comme un bruit de pas .
*****************************************

Epoch 9: train_loss=1.9723
Validation: 0it [00:00, ?it/s]
*****************************************
    SOURCE: « Je n’étais nullement troublé, ajoutait-il, il me semblait que j’accomplissais une cérémonie. »
    TARGET: "I was not in the least anxious," he added, "I felt as though I were performing a ceremony."
 PREDICTED: " I was not at any time ," he added , " he seemed to me that I should like a costume ."
*****************************************

*****************************************
    SOURCE: M. Valenod's hatred was multiplied accordingly.
    TARGET: La haine de M. Valenod redoubla aussi.
 PREDICTED: La haine de M . Valenod était donc .
*****************************************

Epoch 10: train_loss=2.0024
Validation: 0it [00:00, ?it/s]
*****************************************
    SOURCE: He only meant that there was not such a variety of people to be met with in the country as in the town, which you must acknowledge to be true."
    TARGET: Il voulait seulement dire qu’on ne rencontre pas en province une aussi grande variété de gens qu’a Londres et vous devez reconnaître qu’il a raison.
 PREDICTED: Il n ’ a pas voulu qu ’ il n ’ y ait pas une variété de monde dans la campagne que dans la ville , et que vous devez reconnaître .
*****************************************

*****************************************
    SOURCE: As these twenty sous accumulated they would form a nice little sum in four or five years, and when one has money one is ready, eh, for anything that turns up?
    TARGET: Avec ces vingt sous accumulés, on aurait, en quatre ou cinq ans, un magot; et, quand on a de l'argent, on est fort, n'est-ce pas? dans n'importe quelle occasion…
 PREDICTED: A ces vingt sous , on se une jolie somme en quatre ou cinq ans , et quand on a de l ' argent , il est prêt , n ' est - ce que ça se ?
*****************************************

Epoch 11: train_loss=2.1551
Validation: 0it [00:00, ?it/s]
*****************************************
    SOURCE: At three o’clock came two companies of the Guards, one French, the other Swiss.
    TARGET: À trois heures arrivèrent deux compagnies des gardes, l'une française l'autre suisse.
 PREDICTED: À trois heures , il en vint deux compagnies de la Garde , l ' autre .
*****************************************

*****************************************
    SOURCE: But why speak of his friends, his enemies?
    TARGET: Mais à quoi bon nommer ses amis, ses ennemis ?
 PREDICTED: Mais pourquoi parler de ses amis , ses ennemis ?
*****************************************

Epoch 12: train_loss=2.0289
Validation: 0it [00:00, ?it/s]
*****************************************
    SOURCE: There were at that time no lamps in the square before Notre−Dame.
    TARGET: Il n’y avait pas alors de luminaire dans le Parvis de Notre-Dame.
 PREDICTED: Il n ’ y avait pas de temps dans la place que devant Notre - Dame .
*****************************************

*****************************************
    SOURCE: Here it is," replied d’Artagnan, taking it from his finger.
    TARGET: -- Montrez-moi donc cette bague, dit Athos.
 PREDICTED: -- Le voici , reprit d ' Artagnan , qu ' il le donne du doigt .
*****************************************

Epoch 13: train_loss=2.0170
Validation: 0it [00:00, ?it/s]
*****************************************
    SOURCE: They saw the cage disappear and rushed forward, but they had to draw back from the final downfall of the tubbing; the shaft was stopped up and the cage would not come down again.
    TARGET: Ils virent la cage disparaître, ils se précipiterent; mais il leur fallut reculer, sous l'écroulement final du cuvelage: le puits se bouchait, la cage ne redescendrait pas.
 PREDICTED: On voyait disparaître la cage se précipita , on le fit descendre , mais l ' du cuvelage dernier , le puits s ' arreta et la cage ne descendit de marche .
*****************************************

*****************************************
    SOURCE: Mais me ferez-vous le sacrifice immense, ajouta-t-il en riant, de changer le titre sublime de duchesse contre un autre bien inférieur ?
    TARGET: But will you make me the immense sacrifice," he added, laughing, "of exchanging the sublime title of Duchessa for another greatly inferior?
 PREDICTED: But shall I have the last sacrifice , ," he added , laughing , " the sublime Duchessa in another to the other tout cela ?
*****************************************

Epoch 14: train_loss=1.9935
Validation: 0it [00:00, ?it/s]
*****************************************
    SOURCE: Public attention was distracted by the appearance of Monsieur Bournisien, who was going across the market with the holy oil.
    TARGET: L’attention publique fut distraite par l’apparition de M. Bournisien, qui passait sous les halles avec les saintes huiles.
 PREDICTED: L ’ attention fut saisie par l ’ air de M . Bournisien , qui s ’ éloignait du de la sainte huile .
*****************************************

*****************************************
    SOURCE: To the sails were attached strong bolt ropes, and there still remained enough from which to make the halyards, shrouds, and sheets, etc.
    TARGET: Les câbles, les cordages du filet, tout cela était fait d'un filin excellent, dont le marin tira bon parti.
 PREDICTED: A la gare s ' attacher un solide verrou , il restait encore à celui qui pût donner le , les voiles étaient couverts , etc .
*****************************************

Epoch 15: train_loss=1.9746
Validation: 0it [00:00, ?it/s]
*****************************************
    SOURCE: Ayrton then, followed by Pencroft, crossed it with a rapid step, scaring the birds nestled in the holes of the rocks; then, without hesitating, he plunged into the sea, and swam noiselessly in the direction of the ship, in which a few lights had recently appeared, showing her exact situation.
    TARGET: Donc, Ayrton, suivi de Pencroff, le traversa d'un pas rapide, effarouchant les oiseaux nichés dans les trous de roche; puis, sans hésiter, il se jeta à la mer et nagea sans bruit dans la direction du navire, dont quelques lumières, allumées depuis peu, indiquaient alors la situation exacte.
 PREDICTED: Ayrton , suivi de Pencroff , le suivit d ' un pas rapide , les oiseaux se dans les trous de l ' île , puis , sans hésiter , il s ' enfonçait dans la direction du navire , et dans la direction du navire , à quelques feux récemment , dont la situation avait très juste , était faite .
*****************************************

*****************************************
    SOURCE: This caste interest blinded their eyes to the horror of condemning a man to death.
    TARGET: Cet intérêt de caste est venu masquer à leurs yeux l’horreur de condamner à mort.
 PREDICTED: Ce peuple any passion l ’ horreur a la mort .
*****************************************

Epoch 16: train_loss=1.8629
Validation: 0it [00:00, ?it/s]
*****************************************
    SOURCE: La troisième ou quatrième fois qu’elle se trouva au milieu du lac avec tous ces hommes bien choisis, elle fit arrêter le mouvement des rames.
    TARGET: The third or fourth time that she found herself in the middle of the lake with all of these well-chosen men, she stopped the movement of their oars.
 PREDICTED: On the third or three times which she found in the lake with all these men ' s , she made the movement of the .
*****************************************

*****************************************
    SOURCE: "You might say all that to almost any one who you knew lived as a solitary dependent in a great house."
    TARGET: «Vous pourriez dire cela à presque tous ceux qui vivent solitaires et dépendants dans une grande maison.
 PREDICTED: -- Vous avez pu dire tout ce que vous aviez à peine vécu dans une grande maison solitaire .
*****************************************

Epoch 17: train_loss=1.9218
Validation: 0it [00:00, ?it/s]
*****************************************
    SOURCE: The ladies were told all this again, to be sure, and they made themselves merry with it, and every now and then the young ladies, Mr. Mayor's daughters, would come and see me, and ask where the little gentlewoman was, which made me not a little proud of myself. This held a great while, and I was often visited by these young ladies, and sometimes they brought others with them; so that I was known by it almost all over the town.
    TARGET: Tout ceci fut répété aux dames, et elles s'en amusèrent et de temps en temps les filles de M. le maire venaient me voir et demandaient où était la petite dame de qualité, ce qui ne me rendait pas peu fière de moi, d'ailleurs j'avais souvent la visite de ces jeunes dames, et elles en amenaient d'autres avec elles; de sorte que par cela je devins connue presque dans toute la ville.
 PREDICTED: Les dames de M . Rochester étaient trop intelligents pour le voir aussi naturel et ils se firent plaisir à ce sujet ; les jeunes filles de ce temps - ci , les filles de M . le maire , étaient venus de me voir , et me demandant où se trouvait la petite dame de qualité , qui ne me tenait pas un peu fiers de moi ; je fus quelquefois aussi bien que ces jeunes filles , et qu ' elles me le avec d ' autres que je savais que toutes les gens de la ville .
*****************************************

*****************************************
    SOURCE: Elle rend à jamais impossibles tous les projets de fuite ; et Mademoiselle sait mieux que moi que ce n’est pas avec du simple laudanum que l’on veut empoisonner Monseigneur ; elle sait aussi que quelqu’un n’a accordé qu’un mois de délai pour ce crime, et qu’il y a déjà plus d’une semaine que l’ordre fatal a été reçu.
    TARGET: She makes impossible for ever all the plans of escape; and the Signorina knows better than I that it is not with laudanum that they wish to poison Monsignore; she knows, too, that a certain person has granted only a month's delay for that crime, and that already more than a week has gone by since the fatal order was received.
 PREDICTED: She never gave the good fortune for all the plans of ; and Signorina knew that it was impossible with the simple that the poison of Monseigneur ; she knows that the too is only a month of this poison , and that he has retired no longer to the crime , and that there was already one ' s to the fatal order from the fatal order that has been .
*****************************************

Epoch 18: train_loss=1.8536
Validation: 0it [00:00, ?it/s]
*****************************************
    SOURCE: He bit small pieces from it with his teeth, and she chewed them, and endeavoured to swallow them.
    TARGET: Il en coupa de petits morceaux avec les dents, et elle les broyait, s'acharnait a les avaler.
 PREDICTED: Il s ' en alla , les petits morceaux , elle les , tenta de les .
*****************************************

*****************************************
    SOURCE: A big yard, overgrown by weeds where children seemed to have come to play during the long dragging evenings at the end of winter, was hollowed out by the rain.
    TARGET: Une grande cour herbeuse, où des enfants avaient dû venir pendant les longues et lentes soirées de la fin de l’hiver, était ravinée par l’orage.
 PREDICTED: Une grande cour d ’ arbres , par les herbes où les enfants paraissaient jouer aux longues soirées du fond de l ’ hiver , s ’ était par la pluie .
*****************************************

Epoch 19: train_loss=1.7670
Validation: 0it [00:00, ?it/s]
*****************************************
    SOURCE: Having shut the door, M. de Renal seated himself with gravity.
    TARGET: La porte fermée, M. de Rênal s’assit avec gravité.
 PREDICTED: Après avoir fermé la porte , M . de Rênal s ’ assit avec gravité .
*****************************************

*****************************************
    SOURCE: "Have you got it?"
    TARGET: --Vous les avez?
 PREDICTED: -- L ' avez - vous tiré ?
*****************************************

Epoch 20: train_loss=1.9241
Validation: 0it [00:00, ?it/s]
*****************************************
    SOURCE: La Chantefleurie had been a poor creature during the five years since her fall.
    TARGET: Depuis cinq ans qu’elle avait failli, c’était une pauvre créature que la Chantefleurie.
 PREDICTED: La Chantefleurie avait été pauvre créature pendant cinq ans qu ’ elle tomba .
*****************************************

*****************************************
    SOURCE: They regarded her as a distraction that drew them from their bad dreams.
    TARGET: Ils la considéraient comme une distraction qui les tirait de leurs mauvais rêves.
 PREDICTED: Ils la regardaient comme une distraction qu ’ ils la de leurs mauvais rêves .
*****************************************

Epoch 21: train_loss=1.7949
