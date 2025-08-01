import psycopg2
import csv
from transformers import EfficientNetImageProcessor, EfficientNetForImageClassification
from PIL import Image
import torch
from EcoNameTranslator import to_species, to_scientific
import os

preprocessor = EfficientNetImageProcessor.from_pretrained("dennisjooo/Birds-Classifier-EfficientNetB2")
model = EfficientNetForImageClassification.from_pretrained("dennisjooo/Birds-Classifier-EfficientNetB2")

db_config = {
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD'),
    'dbname': os.getenv('DB_NAME'),
    'host': os.getenv('DB_HOST'),
    'port': os.getenv('DB_PORT')
}

bird_labels = {
    "ABBOTTS BABBLER": "0",
    "ABBOTTS BOOBY": "1",
    "ABYSSINIAN GROUND HORNBILL": "2",
    "AFRICAN CROWNED CRANE": "3",
    "AFRICAN EMERALD CUCKOO": "4",
    "AFRICAN FIREFINCH": "5",
    "AFRICAN OYSTER CATCHER": "6",
    "AFRICAN PIED HORNBILL": "7",
    "AFRICAN PYGMY GOOSE": "8",
    "ALBATROSS": "9",
    "ALBERTS TOWHEE": "10",
    "ALEXANDRINE PARAKEET": "11",
    "ALPINE CHOUGH": "12",
    "ALTAMIRA YELLOWTHROAT": "13",
    "AMERICAN AVOCET": "14",
    "AMERICAN BITTERN": "15",
    "AMERICAN COOT": "16",
    "AMERICAN DIPPER": "17",
    "AMERICAN FLAMINGO": "18",
    "AMERICAN GOLDFINCH": "19",
    "AMERICAN KESTREL": "20",
    "AMERICAN PIPIT": "21",
    "AMERICAN REDSTART": "22",
    "AMERICAN ROBIN": "23",
    "AMERICAN WIGEON": "24",
    "AMETHYST WOODSTAR": "25",
    "ANDEAN GOOSE": "26",
    "ANDEAN LAPWING": "27",
    "ANDEAN SISKIN": "28",
    "ANHINGA": "29",
    "ANIANIAU": "30",
    "ANNAS HUMMINGBIRD": "31",
    "ANTBIRD": "32",
    "ANTILLEAN EUPHONIA": "33",
    "APAPANE": "34",
    "APOSTLEBIRD": "35",
    "ARARIPE MANAKIN": "36",
    "ASHY STORM PETREL": "37",
    "ASHY THRUSHBIRD": "38",
    "ASIAN CRESTED IBIS": "39",
    "ASIAN DOLLARD BIRD": "40",
    "ASIAN GREEN BEE EATER": "41",
    "ASIAN OPENBILL STORK": "42",
    "AUCKLAND SHAQ": "43",
    "AUSTRAL CANASTERO": "44",
    "AUSTRALASIAN FIGBIRD": "45",
    "AVADAVAT": "46",
    "AZARAS SPINETAIL": "47",
    "AZURE BREASTED PITTA": "48",
    "AZURE JAY": "49",
    "AZURE TANAGER": "50",
    "AZURE TIT": "51",
    "BAIKAL TEAL": "52",
    "BALD EAGLE": "53",
    "BALD IBIS": "54",
    "BALI STARLING": "55",
    "BALTIMORE ORIOLE": "56",
    "BANANAQUIT": "57",
    "BAND TAILED GUAN": "58",
    "BANDED BROADBILL": "59",
    "BANDED PITA": "60",
    "BANDED STILT": "61",
    "BAR-TAILED GODWIT": "62",
    "BARN OWL": "63",
    "BARN SWALLOW": "64",
    "BARRED PUFFBIRD": "65",
    "BARROWS GOLDENEYE": "66",
    "BAY-BREASTED WARBLER": "67",
    "BEARDED BARBET": "68",
    "BEARDED BELLBIRD": "69",
    "BEARDED REEDLING": "70",
    "BELTED KINGFISHER": "71",
    "BIRD OF PARADISE": "72",
    "BLACK AND YELLOW BROADBILL": "73",
    "BLACK BAZA": "74",
    "BLACK BREASTED PUFFBIRD": "75",
    "BLACK COCKATO": "76",
    "BLACK FACED SPOONBILL": "77",
    "BLACK FRANCOLIN": "78",
    "BLACK HEADED CAIQUE": "79",
    "BLACK NECKED STILT": "80",
    "BLACK SKIMMER": "81",
    "BLACK SWAN": "82",
    "BLACK TAIL CRAKE": "83",
    "BLACK THROATED BUSHTIT": "84",
    "BLACK THROATED HUET": "85",
    "BLACK THROATED WARBLER": "86",
    "BLACK VENTED SHEARWATER": "87",
    "BLACK VULTURE": "88",
    "BLACK-CAPPED CHICKADEE": "89",
    "BLACK-NECKED GREBE": "90",
    "BLACK-THROATED SPARROW": "91",
    "BLACKBURNIAM WARBLER": "92",
    "BLONDE CRESTED WOODPECKER": "93",
    "BLOOD PHEASANT": "94",
    "BLUE COAU": "95",
    "BLUE DACNIS": "96",
    "BLUE GRAY GNATCATCHER": "97",
    "BLUE GROSBEAK": "98",
    "BLUE GROUSE": "99",
    "BLUE HERON": "100",
    "BLUE MALKOHA": "101",
    "BLUE THROATED PIPING GUAN": "102",
    "BLUE THROATED TOUCANET": "103",
    "BOBOLINK": "104",
    "BORNEAN BRISTLEHEAD": "105",
    "BORNEAN LEAFBIRD": "106",
    "BORNEAN PHEASANT": "107",
    "BRANDT CORMARANT": "108",
    "BREWERS BLACKBIRD": "109",
    "BROWN CREPPER": "110",
    "BROWN HEADED COWBIRD": "111",
    "BROWN NOODY": "112",
    "BROWN THRASHER": "113",
    "BUFFLEHEAD": "114",
    "BULWERS PHEASANT": "115",
    "BURCHELLS COURSER": "116",
    "BUSH TURKEY": "117",
    "CAATINGA CACHOLOTE": "118",
    "CABOTS TRAGOPAN": "119",
    "CACTUS WREN": "120",
    "CALIFORNIA CONDOR": "121",
    "CALIFORNIA GULL": "122",
    "CALIFORNIA QUAIL": "123",
    "CAMPO FLICKER": "124",
    "CANARY": "125",
    "CANVASBACK": "126",
    "CAPE GLOSSY STARLING": "127",
    "CAPE LONGCLAW": "128",
    "CAPE MAY WARBLER": "129",
    "CAPE ROCK THRUSH": "130",
    "CAPPED HERON": "131",
    "CAPUCHINBIRD": "132",
    "CARMINE BEE-EATER": "133",
    "CASPIAN TERN": "134",
    "CASSOWARY": "135",
    "CEDAR WAXWING": "136",
    "CERULEAN WARBLER": "137",
    "CHARA DE COLLAR": "138",
    "CHATTERING LORY": "139",
    "CHESTNET BELLIED EUPHONIA": "140",
    "CHESTNUT WINGED CUCKOO": "141",
    "CHINESE BAMBOO PARTRIDGE": "142",
    "CHINESE POND HERON": "143",
    "CHIPPING SPARROW": "144",
    "CHUCAO TAPACULO": "145",
    "CHUKAR PARTRIDGE": "146",
    "CINNAMON ATTILA": "147",
    "CINNAMON FLYCATCHER": "148",
    "CINNAMON TEAL": "149",
    "CLARKS GREBE": "150",
    "CLARKS NUTCRACKER": "151",
    "COCK OF THE  ROCK": "152",
    "COCKATOO": "153",
    "COLLARED ARACARI": "154",
    "COLLARED CRESCENTCHEST": "155",
    "COMMON FIRECREST": "156",
    "COMMON GRACKLE": "157",
    "COMMON HOUSE MARTIN": "158",
    "COMMON IORA": "159",
    "COMMON LOON": "160",
    "COMMON POORWILL": "161",
    "COMMON STARLING": "162",
    "COPPERSMITH BARBET": "163",
    "COPPERY TAILED COUCAL": "164",
    "CRAB PLOVER": "165",
    "CRANE HAWK": "166",
    "CREAM COLORED WOODPECKER": "167",
    "CRESTED AUKLET": "168",
    "CRESTED CARACARA": "169",
    "CRESTED COUA": "170",
    "CRESTED FIREBACK": "171",
    "CRESTED KINGFISHER": "172",
    "CRESTED NUTHATCH": "173",
    "CRESTED OROPENDOLA": "174",
    "CRESTED SERPENT EAGLE": "175",
    "CRESTED SHRIKETIT": "176",
    "CRESTED WOOD PARTRIDGE": "177",
    "CRIMSON CHAT": "178",
    "CRIMSON SUNBIRD": "179",
    "CROW": "180",
    "CUBAN TODY": "181",
    "CUBAN TROGON": "182",
    "CURL CRESTED ARACURI": "183",
    "D-ARNAUDS BARBET": "184",
    "DALMATIAN PELICAN": "185",
    "DARJEELING WOODPECKER": "186",
    "DARK EYED JUNCO": "187",
    "DAURIAN REDSTART": "188",
    "DEMOISELLE CRANE": "189",
    "DOUBLE BARRED FINCH": "190",
    "DOUBLE BRESTED CORMARANT": "191",
    "DOUBLE EYED FIG PARROT": "192",
    "DOWNY WOODPECKER": "193",
    "DUNLIN": "194",
    "DUSKY LORY": "195",
    "DUSKY ROBIN": "196",
    "EARED PITA": "197",
    "EASTERN BLUEBIRD": "198",
    "EASTERN BLUEBONNET": "199",
    "EASTERN GOLDEN WEAVER": "200",
    "EASTERN MEADOWLARK": "201",
    "EASTERN ROSELLA": "202",
    "EASTERN TOWEE": "203",
    "EASTERN WIP POOR WILL": "204",
    "EASTERN YELLOW ROBIN": "205",
    "ECUADORIAN HILLSTAR": "206",
    "EGYPTIAN GOOSE": "207",
    "ELEGANT TROGON": "208",
    "ELLIOTS  PHEASANT": "209",
    "EMERALD TANAGER": "210",
    "EMPEROR PENGUIN": "211",
    "EMU": "212",
    "ENGGANO MYNA": "213",
    "EURASIAN BULLFINCH": "214",
    "EURASIAN GOLDEN ORIOLE": "215",
    "EURASIAN MAGPIE": "216",
    "EUROPEAN GOLDFINCH": "217",
    "EUROPEAN TURTLE DOVE": "218",
    "EVENING GROSBEAK": "219",
    "FAIRY BLUEBIRD": "220",
    "FAIRY PENGUIN": "221",
    "FAIRY TERN": "222",
    "FAN TAILED WIDOW": "223",
    "FASCIATED WREN": "224",
    "FIERY MINIVET": "225",
    "FIORDLAND PENGUIN": "226",
    "FIRE TAILLED MYZORNIS": "227",
    "FLAME BOWERBIRD": "228",
    "FLAME TANAGER": "229",
    "FOREST WAGTAIL": "230",
    "FRIGATE": "231",
    "FRILL BACK PIGEON": "232",
    "GAMBELS QUAIL": "233",
    "GANG GANG COCKATOO": "234",
    "GILA WOODPECKER": "235",
    "GILDED FLICKER": "236",
    "GLOSSY IBIS": "237",
    "GO AWAY BIRD": "238",
    "GOLD WING WARBLER": "239",
    "GOLDEN BOWER BIRD": "240",
    "GOLDEN CHEEKED WARBLER": "241",
    "GOLDEN CHLOROPHONIA": "242",
    "GOLDEN EAGLE": "243",
    "GOLDEN PARAKEET": "244",
    "GOLDEN PHEASANT": "245",
    "GOLDEN PIPIT": "246",
    "GOULDIAN FINCH": "247",
    "GRANDALA": "248",
    "GRAY CATBIRD": "249",
    "GRAY KINGBIRD": "250",
    "GRAY PARTRIDGE": "251",
    "GREAT ARGUS": "252",
    "GREAT GRAY OWL": "253",
    "GREAT JACAMAR": "254",
    "GREAT KISKADEE": "255",
    "GREAT POTOO": "256",
    "GREAT TINAMOU": "257",
    "GREAT XENOPS": "258",
    "GREATER PEWEE": "259",
    "GREATER PRAIRIE CHICKEN": "260",
    "GREATOR SAGE GROUSE": "261",
    "GREEN BROADBILL": "262",
    "GREEN JAY": "263",
    "GREEN MAGPIE": "264",
    "GREEN WINGED DOVE": "265",
    "GREY CUCKOOSHRIKE": "266",
    "GREY HEADED CHACHALACA": "267",
    "GREY HEADED FISH EAGLE": "268",
    "GREY PLOVER": "269",
    "GROVED BILLED ANI": "270",
    "GUINEA TURACO": "271",
    "GUINEAFOWL": "272",
    "GURNEYS PITTA": "273",
    "GYRFALCON": "274",
    "HAMERKOP": "275",
    "HARLEQUIN DUCK": "276",
    "HARLEQUIN QUAIL": "277",
    "HARPY EAGLE": "278",
    "HAWAIIAN GOOSE": "279",
    "HAWFINCH": "280",
    "HELMET VANGA": "281",
    "HEPATIC TANAGER": "282",
    "HIMALAYAN BLUETAIL": "283",
    "HIMALAYAN MONAL": "284",
    "HOATZIN": "285",
    "HOODED MERGANSER": "286",
    "HOOPOES": "287",
    "HORNED GUAN": "288",
    "HORNED LARK": "289",
    "HORNED SUNGEM": "290",
    "HOUSE FINCH": "291",
    "HOUSE SPARROW": "292",
    "HYACINTH MACAW": "293",
    "IBERIAN MAGPIE": "294",
    "IBISBILL": "295",
    "IMPERIAL SHAQ": "296",
    "INCA TERN": "297",
    "INDIAN BUSTARD": "298",
    "INDIAN PITTA": "299",
    "INDIAN ROLLER": "300",
    "INDIAN VULTURE": "301",
    "INDIGO BUNTING": "302",
    "INDIGO FLYCATCHER": "303",
    "INLAND DOTTEREL": "304",
    "IVORY BILLED ARACARI": "305",
    "IVORY GULL": "306",
    "IWI": "307",
    "JABIRU": "308",
    "JACK SNIPE": "309",
    "JACOBIN PIGEON": "310",
    "JANDAYA PARAKEET": "311",
    "JAPANESE ROBIN": "312",
    "JAVA SPARROW": "313",
    "JOCOTOCO ANTPITTA": "314",
    "KAGU": "315",
    "KAKAPO": "316",
    "KILLDEAR": "317",
    "KING EIDER": "318",
    "KING VULTURE": "319",
    "KIWI": "320",
    "KNOB BILLED DUCK": "321",
    "KOOKABURRA": "322",
    "LARK BUNTING": "323",
    "LAUGHING GULL": "324",
    "LAZULI BUNTING": "325",
    "LESSER ADJUTANT": "326",
    "LILAC ROLLER": "327",
    "LIMPKIN": "328",
    "LITTLE AUK": "329",
    "LOGGERHEAD SHRIKE": "330",
    "LONG-EARED OWL": "331",
    "LOONEY BIRDS": "332",
    "LUCIFER HUMMINGBIRD": "333",
    "MAGPIE GOOSE": "334",
    "MALABAR HORNBILL": "335",
    "MALACHITE KINGFISHER": "336",
    "MALAGASY WHITE EYE": "337",
    "MALEO": "338",
    "MALLARD DUCK": "339",
    "MANDRIN DUCK": "340",
    "MANGROVE CUCKOO": "341",
    "MARABOU STORK": "342",
    "MASKED BOBWHITE": "343",
    "MASKED BOOBY": "344",
    "MASKED LAPWING": "345",
    "MCKAYS BUNTING": "346",
    "MERLIN": "347",
    "MIKADO  PHEASANT": "348",
    "MILITARY MACAW": "349",
    "MOURNING DOVE": "350",
    "MYNA": "351",
    "NICOBAR PIGEON": "352",
    "NOISY FRIARBIRD": "353",
    "NORTHERN BEARDLESS TYRANNULET": "354",
    "NORTHERN CARDINAL": "355",
    "NORTHERN FLICKER": "356",
    "NORTHERN FULMAR": "357",
    "NORTHERN GANNET": "358",
    "NORTHERN GOSHAWK": "359",
    "NORTHERN JACANA": "360",
    "NORTHERN MOCKINGBIRD": "361",
    "NORTHERN PARULA": "362",
    "NORTHERN RED BISHOP": "363",
    "NORTHERN SHOVELER": "364",
    "OCELLATED TURKEY": "365",
    "OILBIRD": "366",
    "OKINAWA RAIL": "367",
    "ORANGE BREASTED TROGON": "368",
    "ORANGE BRESTED BUNTING": "369",
    "ORIENTAL BAY OWL": "370",
    "ORNATE HAWK EAGLE": "371",
    "OSPREY": "372",
    "OSTRICH": "373",
    "OVENBIRD": "374",
    "OYSTER CATCHER": "375",
    "PAINTED BUNTING": "376",
    "PALILA": "377",
    "PALM NUT VULTURE": "378",
    "PARADISE TANAGER": "379",
    "PARAKETT  AUKLET": "380",
    "PARUS MAJOR": "381",
    "PATAGONIAN SIERRA FINCH": "382",
    "PEACOCK": "383",
    "PEREGRINE FALCON": "384",
    "PHAINOPEPLA": "385",
    "PHILIPPINE EAGLE": "386",
    "PINK ROBIN": "387",
    "PLUSH CRESTED JAY": "388",
    "POMARINE JAEGER": "389",
    "PUFFIN": "390",
    "PUNA TEAL": "391",
    "PURPLE FINCH": "392",
    "PURPLE GALLINULE": "393",
    "PURPLE MARTIN": "394",
    "PURPLE SWAMPHEN": "395",
    "PYGMY KINGFISHER": "396",
    "PYRRHULOXIA": "397",
    "QUETZAL": "398",
    "RAINBOW LORIKEET": "399",
    "RAZORBILL": "400",
    "RED BEARDED BEE EATER": "401",
    "RED BELLIED PITTA": "402",
    "RED BILLED TROPICBIRD": "403",
    "RED BROWED FINCH": "404",
    "RED CROSSBILL": "405",
    "RED FACED CORMORANT": "406",
    "RED FACED WARBLER": "407",
    "RED FODY": "408",
    "RED HEADED DUCK": "409",
    "RED HEADED WOODPECKER": "410",
    "RED KNOT": "411",
    "RED LEGGED HONEYCREEPER": "412",
    "RED NAPED TROGON": "413",
    "RED SHOULDERED HAWK": "414",
    "RED TAILED HAWK": "415",
    "RED TAILED THRUSH": "416",
    "RED WINGED BLACKBIRD": "417",
    "RED WISKERED BULBUL": "418",
    "REGENT BOWERBIRD": "419",
    "RING-NECKED PHEASANT": "420",
    "ROADRUNNER": "421",
    "ROCK DOVE": "422",
    "ROSE BREASTED COCKATOO": "423",
    "ROSE BREASTED GROSBEAK": "424",
    "ROSEATE SPOONBILL": "425",
    "ROSY FACED LOVEBIRD": "426",
    "ROUGH LEG BUZZARD": "427",
    "ROYAL FLYCATCHER": "428",
    "RUBY CROWNED KINGLET": "429",
    "RUBY THROATED HUMMINGBIRD": "430",
    "RUDDY SHELDUCK": "431",
    "RUDY KINGFISHER": "432",
    "RUFOUS KINGFISHER": "433",
    "RUFOUS TREPE": "434",
    "RUFUOS MOTMOT": "435",
    "SAMATRAN THRUSH": "436",
    "SAND MARTIN": "437",
    "SANDHILL CRANE": "438",
    "SATYR TRAGOPAN": "439",
    "SAYS PHOEBE": "440",
    "SCARLET CROWNED FRUIT DOVE": "441",
    "SCARLET FACED LIOCICHLA": "442",
    "SCARLET IBIS": "443",
    "SCARLET MACAW": "444",
    "SCARLET TANAGER": "445",
    "SHOEBILL": "446",
    "SHORT BILLED DOWITCHER": "447",
    "SMITHS LONGSPUR": "448",
    "SNOW GOOSE": "449",
    "SNOW PARTRIDGE": "450",
    "SNOWY EGRET": "451",
    "SNOWY OWL": "452",
    "SNOWY PLOVER": "453",
    "SNOWY SHEATHBILL": "454",
    "SORA": "455",
    "SPANGLED COTINGA": "456",
    "SPLENDID WREN": "457",
    "SPOON BILED SANDPIPER": "458",
    "SPOTTED CATBIRD": "459",
    "SPOTTED WHISTLING DUCK": "460",
    "SQUACCO HERON": "461",
    "SRI LANKA BLUE MAGPIE": "462",
    "STEAMER DUCK": "463",
    "STORK BILLED KINGFISHER": "464",
    "STRIATED CARACARA": "465",
    "STRIPED OWL": "466",
    "STRIPPED MANAKIN": "467",
    "STRIPPED SWALLOW": "468",
    "SUNBITTERN": "469",
    "SUPERB STARLING": "470",
    "SURF SCOTER": "471",
    "SWINHOES PHEASANT": "472",
    "TAILORBIRD": "473",
    "TAIWAN MAGPIE": "474",
    "TAKAHE": "475",
    "TASMANIAN HEN": "476",
    "TAWNY FROGMOUTH": "477",
    "TEAL DUCK": "478",
    "TIT MOUSE": "479",
    "TOUCHAN": "480",
    "TOWNSENDS WARBLER": "481",
    "TREE SWALLOW": "482",
    "TRICOLORED BLACKBIRD": "483",
    "TROPICAL KINGBIRD": "484",
    "TRUMPTER SWAN": "485",
    "TURKEY VULTURE": "486",
    "TURQUOISE MOTMOT": "487",
    "UMBRELLA BIRD": "488",
    "VARIED THRUSH": "489",
    "VEERY": "490",
    "VENEZUELIAN TROUPIAL": "491",
    "VERDIN": "492",
    "VERMILION FLYCATHER": "493",
    "VICTORIA CROWNED PIGEON": "494",
    "VIOLET BACKED STARLING": "495",
    "VIOLET CUCKOO": "496",
    "VIOLET GREEN SWALLOW": "497",
    "VIOLET TURACO": "498",
    "VISAYAN HORNBILL": "499",
    "VULTURINE GUINEAFOWL": "500",
    "WALL CREAPER": "501",
    "WATTLED CURASSOW": "502",
    "WATTLED LAPWING": "503",
    "WHIMBREL": "504",
    "WHITE BREASTED WATERHEN": "505",
    "WHITE BROWED CRAKE": "506",
    "WHITE CHEEKED TURACO": "507",
    "WHITE CRESTED HORNBILL": "508",
    "WHITE EARED HUMMINGBIRD": "509",
    "WHITE NECKED RAVEN": "510",
    "WHITE TAILED TROPIC": "511",
    "WHITE THROATED BEE EATER": "512",
    "WILD TURKEY": "513",
    "WILLOW PTARMIGAN": "514",
    "WILSONS BIRD OF PARADISE": "515",
    "WOOD DUCK": "516",
    "WOOD THRUSH": "517",
    "WOODLAND KINGFISHER": "518",
    "WRENTIT": "519",
    "YELLOW BELLIED FLOWERPECKER": "520",
    "YELLOW BREASTED CHAT": "521",
    "YELLOW CACIQUE": "522",
    "YELLOW HEADED BLACKBIRD": "523",
    "ZEBRA DOVE": "524"
}

known_species = set()

def get_db_conn():
    try:
        connection = psycopg2.connect(**db_config)
        return connection
    except Exception as e:
        print(f"get_db_connection: {e}")


def refresh_known_species():
    global known_species
    connection = get_db_conn()
    cursor = connection.cursor()
    cursor.execute("SELECT DISTINCT species_name FROM birds;")
    results = cursor.fetchall()
    known_species.update(result[0].lower() for result in results)
    cursor.close()
    connection.close()

def setup_cache():
    refresh_known_species()
    connection = get_db_conn()
    cursor = connection.cursor()
    counter = 0
    batch = []
    query = '''INSERT INTO synonyms (label_id, synonym) VALUES (%s, %s);'''
    print(f"We have {len(bird_labels.keys())} labels retrived.")
    for (label, id) in bird_labels.items():
        label = label.lower()
        label_id = int(id) + 1 # because table label counts from 1 not from 0
        new_label = None
        steps = 0
        while new_label is None and steps < 10:
            new_label = retrieve_known_species(label)
            steps = steps + 1
            if steps == 10 and new_label is None:
                print(f"Did not find label for {label}.")
        counter = counter + 1
        if new_label is not None and new_label not in ["null", "Null", "NULL", ""]:
            print(f"Adding new label: {new_label}")
            batch.append((label_id, new_label))
            cursor.executemany(query, batch)
            connection.commit()
            batch = []
        if counter % 5 == 0:
            print(f"Processed {counter} synonyms.")
    cursor.close()
    connection.close()

def retrieve_known_species(label):
    label = label.lower()
    split_label = label.split(" ")
    split_label.append(label)
    split_label = list(set(split_label))
    #print(f"Split_label = {split_label}")
    label_synonyms = {}
    all_synonyms = to_scientific(split_label)
    for split in split_label:
        try:
            label_synonyms[split] = all_synonyms.get(split, ["", []])[1]
            #print(f"Synonyms for {split}: {label_synonyms[split]}")
        except Exception as e:
            #print(f"Error synonyms extraction: {e}")
            continue
    #refresh_known_species()
    #print("Sorting keys now")
    sorted_keys = sorted(label_synonyms.keys(), key=lambda k: -k.count(" "))
    for key in sorted_keys:
        for val in label_synonyms[key]:
            #print(f"Check if key matches: {val} and count {len(known_species)}")
            if val.lower() in known_species:
                print(f"HIT: {val.lower()}")
                return val
    return None

def insert_bird_labels():
    connection = psycopg2.connect(**db_config)
    cursor = connection.cursor()
    query = '''INSERT INTO labels (label) VALUES (%s);'''
    batch = [(key,) for key in bird_labels.keys()]
    cursor.executemany(query, batch)
    connection.commit()
    cursor.close()
    connection.close()

def createMaterializedView():
    connection = psycopg2.connect(**db_config)
    cursor = connection.cursor()

    cursor.execute('''
        CREATE MATERIALIZED VIEW birds_materialized AS
        SELECT species_name, canton, SUM(counter) AS total_count, year_number
        FROM birds
        GROUP BY species_name, canton, year_number;
    ''')
    connection.commit()
    cursor.close()
    connection.close()

def getCount():
    connection = psycopg2.connect(**db_config)
    cursor = connection.cursor()
    cursor.execute("SELECT COUNT(*) FROM synonyms;")
    print(cursor.fetchall())
    cursor.close()
    connection.close()


def integrate():
    try:
        connection = psycopg2.connect(**db_config)
        cursor = connection.cursor()

        with open('./vogelwarte.csv', encoding='utf-8') as file:
            reader = csv.DictReader(file, delimiter=';')

            data_to_insert = []
            for row in reader:
                # Select only the columns you want
                year = int(row['YearName']) if row['YearName'].strip() else 0
                trivial = row['TrivialName'].strip().capitalize()
                species = row['SpeciesName'].strip().capitalize()
                city = row['City'].strip().capitalize()
                canton = row['Kanton'].strip().upper()
                x = float(row['CoordX']) if row['CoordX'].strip() else 0.0
                y = float(row['CoordY']) if row['CoordY'].strip() else 0.0
                count = int(row['nNachweis']) if row['nNachweis'].strip() else 1

                data_to_insert.append((trivial, species, city, canton, x, y, count, year))

            insert_query = """
                INSERT INTO birds (trivial_name, species_name, city, canton, x, y, counter, year_number)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """

            cursor.executemany(insert_query, data_to_insert)
            connection.commit()
            print("Data inserted successfully.")

    except Exception as e:
        print("Error:", e)
        print("Are you sure, that the file 'vogelwarte.csv' exists?")
    finally:
        cursor.close()
        connection.close()

def drop():
    connection = psycopg2.connect(**db_config)
    cursor = connection.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS birds (
            id SERIAL PRIMARY KEY,
            trivial_name TEXT,
            species_name TEXT,
            city TEXT,
            canton TEXT,
            x DECIMAL(18, 15),
            y DECIMAL(18, 15),
            counter INTEGER,
            year_number INTEGER
        );
    ''')
    connection.commit()
    cursor.close()
    connection.close()

def create_cache():
    connection = psycopg2.connect(**db_config)
    cursor = connection.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS labels (
            id SERIAL PRIMARY KEY,
            label TEXT UNIQUE NOT NULL
        );
    ''')
    connection.commit()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS synonyms (
            label_id INTEGER REFERENCES labels(id) ON DELETE CASCADE,
            synonym TEXT NOT NULL,
            PRIMARY KEY (label_id, synonym)
        );
    ''')
    connection.commit()
    cursor.close()
    connection.close()

def performQuery():
    connection = psycopg2.connect(**db_config)
    cursor = connection.cursor()
    cursor.execute('''
        SELECT COUNT(*)
        FROM labels;
    ''')
    results = cursor.fetchall()
    print(results)
    cursor.close()
    connection.close()

if __name__ == "__main__":
    #getCount()
    integrate()
    setup_cache()
    #insert_bird_labels()