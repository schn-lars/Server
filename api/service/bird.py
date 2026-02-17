from PIL import Image
import io
from transformers import EfficientNetImageProcessor, EfficientNetForImageClassification
import torch
from sqlalchemy.orm import Session
from sqlalchemy import func
from entities import Label, Synonym, BirdsMaterialized
from EcoNameTranslator import to_species, to_scientific
import matplotlib.pyplot as plt

preprocessor = EfficientNetImageProcessor.from_pretrained("dennisjooo/Birds-Classifier-EfficientNetB2")
model = EfficientNetForImageClassification.from_pretrained("dennisjooo/Birds-Classifier-EfficientNetB2")

async def classify_bird(file, db: Session):
    print("Classify_bird")
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")

    inputs = preprocessor(img, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits

    predicted_label = logits.argmax(-1).item()
    label = model.config.id2label[predicted_label]
    normalized_label = label.strip().lower()

    label_obj = db.query(Synonym.synonym).join(Label, Label.id == Synonym.label_id).where(Label.label == label).first()
    if label_obj is None:
        return normalized_label
    return label_obj

def create_bird_plot(canton, bird_name, language, db: Session):
    try:
        result = None
        if canton is None:
            command = '''
                SELECT year_number, SUM(total_count)
                FROM birds_materialized
                WHERE species_name = %s
                GROUP BY year_number
                LIMIT 10;
            '''
            result = db\
                .query(
                    BirdsMaterialized.year_number,
                func.sum(BirdsMaterialized.total_count).label("total_count"))\
                .where(BirdsMaterialized.species_name == bird_name)\
                .group_by(BirdsMaterialized.year_number)\
                .limit(limit=10)\
                .all()
        else:
            command = '''
                SELECT year_number, SUM(total_count)
                FROM birds_materialized
                WHERE canton = %s
                AND species_name = %s
                GROUP BY year_number
                LIMIT 10;
            '''
            result = db\
                .query(
                    BirdsMaterialized.year_number,
                    func.sum(BirdsMaterialized.total_count).label("total_count"))\
                .where(BirdsMaterialized.species_name == bird_name, BirdsMaterialized.canton == canton)\
                .group_by(BirdsMaterialized.year_number)\
                .limit(limit=10)\
                .all()
        if not result:
            command = '''
                SELECT year_number, SUM(total_count)
                FROM birds_materialized
                WHERE species_name = %s
                GROUP BY year_number
                LIMIT 10;
            '''
            result = db.query(BirdsMaterialized)\
                .where(BirdsMaterialized.species_name == bird_name)\
                .group_by(BirdsMaterialized.year_number)\
                .limit(limit=10)\
                .all()
            if result:
                year_to_count = {year: count for year, count in result}
                max_year = max(year_to_count.keys())
                years = list(range(max_year - 9, max_year + 1))
                filled_results = [(year, year_to_count.get(year, 0)) for year in years]
                years, counts = zip(*filled_results)

                fig, ax = plt.subplots()
                ax.plot(years, counts, marker='o')
                ax.set_title(
                    f"Occurrences of {bird_name} in Switzerland" if language == "ENG" else f"Sichtungen von {bird_name} in der Schweiz")
                ax.set_xlabel("Year" if language == "ENG" else "Jahr")
                ax.set_ylabel("Count" if language == "ENG" else "Anzahl")
                fig.autofmt_xdate()

                buf = io.BytesIO()
                plt.savefig(buf, format="png")
                buf.seek(0)
                plt.close(fig)
                return buf
            else:
                print(f"No data found for {bird_name} and {canton}.")
                return None
        else:
            year_to_count = {year: count for year, count in result}
            max_year = max(year_to_count.keys())
            years = list(range(max_year - 9, max_year + 1))
            filled_results = [(year, year_to_count.get(year, 0)) for year in years]
            years, counts = zip(*filled_results)

            fig, ax = plt.subplots()
            ax.plot(years, counts, marker='o')
            ax.set_title(f"Occurrences of {bird_name} in {canton}" if language == "ENG" else f"Sichtungen von {bird_name} in {canton}")
            ax.set_xlabel("Year" if language == "ENG" else "Jahr")
            ax.set_ylabel("Count" if language == "ENG" else "Anzahl")
            fig.autofmt_xdate()

            buf = io.BytesIO()
            plt.savefig(buf, format="png")
            buf.seek(0)
            plt.close(fig)
            return buf
    except Exception as e:
        print(f"Error in create_bird_plot: {e}")
        raise