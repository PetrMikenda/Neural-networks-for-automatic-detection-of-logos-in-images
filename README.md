# Skripty pro automatickou detekci log v obraze

Tento repozitář obsahuje sadu sjednocených trénovacích a benchmarkovacích skriptů pro modely rodiny YOLO a transformer architekturu RF-DETR/RT-DETR. Skripty byly vytvořeny primárně pro experimentální část bakalářské práce zabývající se porovnáním těchto architektur na datasetech FlickrLogos-32 a QMUL-OpenLogo.

## Odlišnosti skriptů

Jádro všech trénovacích skriptů je sjednocené (obsahuje logování přes Weights & Biases, uzamčení generátorů náhody pro reprodukovatelnost a optimalizaci paměti). Skripty jsou tematicky rozděleny do tří skupin podle testovaných hypotéz, přičemž se mírně liší svými parametry:

### 1. Skripty pro testování kapacity modelů
Základní trénovací skripty, kde je klíčovým parametrem dynamicky nastavitelná velikost trénovací dávky (`batch_size`). 
* **Důvod odlišnosti:** Při přechodu z jednoduššího datasetu FlickrLogos-32 na komplexnější QMUL-OpenLogo docházelo u větších modelů k narážení na hardwarové limity VRAM. Skripty proto umožňují flexibilně snížit `batch_size` (z 16 na 8) přímo z příkazové řádky, aby nedocházelo k chybám typu *Out of Memory* (OOM).

### 2. Skripty pro testování rozlišení
Skripty navržené pro testování schopnosti modelů detekovat malá loga při zvýšeném vstupním rozlišení.
* **Důvod odlišnosti:** Vstupní rozlišení obrazu je zde staticky zafixováno na hodnotu `1024 px`

### 3. Skripty pro testování augmentací
Skripty zkoumající vliv datových deformací na přesnost detekce a tvarovou stálost log.
* **Důvod odlišnosti:** Tyto skripty obsahují navíc definované augmentační profily. K dispozici jsou tři pevně dané profily:
  * `no_aug` (striktně vypnuté všechny výchozí augmentace)
  * `heavy_aug` (agresivní geometrické a barevné transformace)
  * `logo_optimized` (doménově specifický profil zachovávající identitu značky)

---

## Měření VRAM

Skripty určené k měření latence (FPS) a spotřeby VRAM jsou pro všechny modely sjednocené, aby zajišťovaly objektivní a metodologicky čisté srovnání:
* Ostrému měření předchází 20 iterací pro zahřátí GPU (warm-up).
* U obou architektur je měřena *end-to-end* latence, která spravedlivě zahrnuje i post-processing

