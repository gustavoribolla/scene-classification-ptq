# QuintoAndar External Evaluation

- Manifest: `/Users/queca/Library/Mobile Documents/com~apple~CloudDocs/cv project/scene-classification-ptq/data/external/quintoandar/manifest.csv`
- Quantized model: `/Users/queca/Library/Mobile Documents/com~apple~CloudDocs/cv project/scene-classification-ptq/results/quantized_demo/places365_resnet50_int8_torchscript.pt`
- Samples: 810
- Top-1 mapped accuracy: 52.84%
- Top-5 mapped accuracy: 78.64%

## Label Mapping
- `academia` -> g / gymnasium / indoor, m / martial arts gym
- `area_externa` -> p / patio, p / porch, c / courtyard, y / yard, l / lawn, d / driveway, h / house, b / building facade, a / apartment building / outdoor, b / balcony / exterior
- `area_servico` -> u / utility room, s / storage room, l / laundromat
- `banheiro` -> b / bathroom, s / shower
- `churrasqueira` -> p / patio, c / courtyard, d / dining room, k / kitchen
- `closet` -> c / closet, d / dressing room
- `corredor` -> c / corridor
- `cozinha` -> k / kitchen, p / pantry, r / restaurant kitchen
- `escritorio` -> h / home office, o / office
- `garagem` -> g / garage / indoor, g / garage / outdoor, p / parking garage / indoor, p / parking garage / outdoor, d / driveway
- `jardim` -> y / yard, l / lawn, c / courtyard, f / formal garden, t / topiary garden
- `piscina` -> s / swimming pool / indoor, s / swimming pool / outdoor
- `quarto` -> b / bedroom, b / bedchamber, d / dorm room
- `sala` -> l / living room, d / dining room, t / television room
- `varanda` -> b / balcony / interior, b / balcony / exterior, p / patio, p / porch

## Per Label

### academia
- Count: 3
- Top-1 mapped accuracy: 66.67%
- Top-5 mapped accuracy: 100.00%
- Most common top-1 predictions:
  - `g / gymnasium / indoor`: 2
  - `e / elevator lobby`: 1

### area_externa
- Count: 31
- Top-1 mapped accuracy: 19.35%
- Top-5 mapped accuracy: 51.61%
- Most common top-1 predictions:
  - `j / jacuzzi / indoor`: 4
  - `m / mezzanine`: 3
  - `b / balcony / exterior`: 2
  - `g / garage / outdoor`: 2
  - `m / medina`: 2

### area_servico
- Count: 39
- Top-1 mapped accuracy: 17.95%
- Top-5 mapped accuracy: 56.41%
- Most common top-1 predictions:
  - `b / bathroom`: 8
  - `u / utility room`: 7
  - `g / garage / indoor`: 3
  - `a / artists loft`: 3
  - `c / corridor`: 3

### banheiro
- Count: 137
- Top-1 mapped accuracy: 92.70%
- Top-5 mapped accuracy: 97.81%
- Most common top-1 predictions:
  - `b / bathroom`: 88
  - `s / shower`: 39
  - `e / elevator / door`: 3
  - `b / burial chamber`: 1
  - `j / jail cell`: 1

### banheiro_churrasqueira
- Count: 3
- Top-1 mapped accuracy: 100.00%
- Top-5 mapped accuracy: 100.00%
- Most common top-1 predictions:
  - `b / bathroom`: 3

### churrasqueira
- Count: 12
- Top-1 mapped accuracy: 8.33%
- Top-5 mapped accuracy: 50.00%
- Most common top-1 predictions:
  - `a / attic`: 3
  - `a / alcove`: 3
  - `b / burial chamber`: 1
  - `p / porch`: 1
  - `g / gazebo / exterior`: 1

### corredor
- Count: 14
- Top-1 mapped accuracy: 57.14%
- Top-5 mapped accuracy: 78.57%
- Most common top-1 predictions:
  - `c / corridor`: 8
  - `e / elevator / door`: 2
  - `e / entrance hall`: 2
  - `p / porch`: 1
  - `a / artists loft`: 1

### corredor_garagem
- Count: 1
- Top-1 mapped accuracy: 100.00%
- Top-5 mapped accuracy: 100.00%
- Most common top-1 predictions:
  - `c / corridor`: 1

### cozinha
- Count: 67
- Top-1 mapped accuracy: 62.69%
- Top-5 mapped accuracy: 89.55%
- Most common top-1 predictions:
  - `k / kitchen`: 42
  - `w / wet bar`: 5
  - `g / galley`: 4
  - `y / youth hostel`: 3
  - `r / reception`: 2

### cozinha_area_servico
- Count: 6
- Top-1 mapped accuracy: 66.67%
- Top-5 mapped accuracy: 100.00%
- Most common top-1 predictions:
  - `k / kitchen`: 2
  - `c / clean room`: 2
  - `u / utility room`: 2

### escritorio
- Count: 8
- Top-1 mapped accuracy: 12.50%
- Top-5 mapped accuracy: 37.50%
- Most common top-1 predictions:
  - `u / utility room`: 2
  - `l / locker room`: 1
  - `o / office`: 1
  - `b / basement`: 1
  - `b / bathroom`: 1

### garagem
- Count: 32
- Top-1 mapped accuracy: 50.00%
- Top-5 mapped accuracy: 78.12%
- Most common top-1 predictions:
  - `g / garage / indoor`: 14
  - `c / corridor`: 4
  - `b / basement`: 3
  - `a / artists loft`: 2
  - `m / mezzanine`: 2

### jardim
- Count: 11
- Top-1 mapped accuracy: 0.00%
- Top-5 mapped accuracy: 54.55%
- Most common top-1 predictions:
  - `s / swimming pool / indoor`: 3
  - `e / elevator lobby`: 2
  - `o / orchard`: 1
  - `r / roof garden`: 1
  - `w / waiting room`: 1

### piscina
- Count: 4
- Top-1 mapped accuracy: 25.00%
- Top-5 mapped accuracy: 50.00%
- Most common top-1 predictions:
  - `r / railroad track`: 1
  - `c / construction site`: 1
  - `j / jacuzzi / indoor`: 1
  - `s / swimming pool / indoor`: 1

### quarto
- Count: 201
- Top-1 mapped accuracy: 37.31%
- Top-5 mapped accuracy: 72.14%
- Most common top-1 predictions:
  - `b / bedroom`: 62
  - `y / youth hostel`: 25
  - `d / dorm room`: 13
  - `c / closet`: 12
  - `a / alcove`: 9

### quarto_banheiro
- Count: 54
- Top-1 mapped accuracy: 79.63%
- Top-5 mapped accuracy: 96.30%
- Most common top-1 predictions:
  - `b / bathroom`: 31
  - `s / shower`: 12
  - `j / jacuzzi / indoor`: 2
  - `l / laundromat`: 1
  - `l / locker room`: 1

### quarto_closet
- Count: 4
- Top-1 mapped accuracy: 100.00%
- Top-5 mapped accuracy: 100.00%
- Most common top-1 predictions:
  - `c / closet`: 4

### quarto_varanda
- Count: 20
- Top-1 mapped accuracy: 35.00%
- Top-5 mapped accuracy: 50.00%
- Most common top-1 predictions:
  - `b / balcony / interior`: 7
  - `c / closet`: 4
  - `c / construction site`: 2
  - `v / village`: 1
  - `r / roof garden`: 1

### sala
- Count: 130
- Top-1 mapped accuracy: 45.38%
- Top-5 mapped accuracy: 76.92%
- Most common top-1 predictions:
  - `d / dining room`: 25
  - `t / television room`: 20
  - `l / living room`: 14
  - `w / waiting room`: 13
  - `a / artists loft`: 10

### sala_varanda
- Count: 12
- Top-1 mapped accuracy: 75.00%
- Top-5 mapped accuracy: 100.00%
- Most common top-1 predictions:
  - `b / balcony / interior`: 9
  - `r / roof garden`: 1
  - `s / staircase`: 1
  - `b / bow window / indoor`: 1

### varanda
- Count: 16
- Top-1 mapped accuracy: 50.00%
- Top-5 mapped accuracy: 68.75%
- Most common top-1 predictions:
  - `b / balcony / interior`: 5
  - `b / balcony / exterior`: 2
  - `f / fire escape`: 2
  - `b / beach house`: 1
  - `m / medina`: 1

### varanda_area_servico
- Count: 5
- Top-1 mapped accuracy: 80.00%
- Top-5 mapped accuracy: 100.00%
- Most common top-1 predictions:
  - `u / utility room`: 4
  - `k / kitchen`: 1
