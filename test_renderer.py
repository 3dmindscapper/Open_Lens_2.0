"""
test_renderer.py — Test the renderer with mock OCR blocks against the test document.

Simulates what dots.ocr would return for the French certificate document,
runs inpaint + render, and saves the result for visual inspection.
No model loading required.
"""
import os
import sys
from PIL import Image

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from pipeline.inpaint import erase_text_blocks
from pipeline.renderer import render_translations

# ── Mock OCR blocks (simulating dots.ocr output for the French certificate) ──
# These approximate the bounding boxes and text from the test document.
# Coordinates are in pixels at 150 DPI (the image is ~1240x1754 px).

MOCK_BLOCKS = [
    {
        "text": "Greffe du Tribunal de Commercede Créteil\nIMMEUBLE LE PASCAL\nCENTRE COMMERCIAL DE CRETEIL SOLEIL\n94000 CRETEIL CEDEX",
        "translated": "Registry of the Commercial Trade Tribunal\nBUILDING THE PASCAL\nSOLEIL CRETEIL TRADE CENTER\n94000 CRETEIL CEDEX",
        "bbox": (30, 25, 360, 80),
        "category": "Text",
        "lang": "fr",
    },
    {
        "text": "N° de gestion 2021B03153",
        "translated": "Management No. 2021B03153",
        "bbox": (30, 85, 200, 100),
        "category": "Text",
        "lang": "fr",
    },
    {
        "text": "*Extrait Kbis*",
        "translated": "Extract Kbis",
        "bbox": (200, 95, 370, 115),
        "category": "Title",
        "lang": "fr",
    },
    {
        "text": "**EXTRAIT D'IMMATRICULATION PRINCIPALE AU REGISTRE DU COMMERCE ET DES SOCIETES**\nà jour au 11 septembre 2024",
        "translated": "**MAIN IMMATRICULATION EXTRACT IN THE TRADE AND SOCIETY REGISTER**\nas of 11 September 2024",
        "bbox": (70, 120, 560, 155),
        "category": "Title",
        "lang": "fr",
    },
    {
        "text": "**IDENTIFICATION DE LA PERSONNE MORALE**",
        "translated": "**IDENTIFICATION OF THE MORAL PERSON**",
        "bbox": (30, 160, 560, 175),
        "category": "Section-header",
        "lang": "fr",
    },
    {
        "text": "Immatriculation au RCS, numéro\nDate d'immatriculation\nImmatriculation radiée le",
        "translated": "Registration at SCN, number\nDate of registration\nRegistration deleted on",
        "bbox": (30, 180, 200, 230),
        "category": "Text",
        "lang": "fr",
    },
    {
        "text": "***Redacted***\n10/04/2008\n27/06/2024",
        "translated": "***Redacted***\n10/04/2008\n27/06/2024",
        "bbox": (210, 180, 400, 230),
        "category": "Text",
        "lang": "fr",
    },
    {
        "text": "Dénomination ou raison sociale\nForme juridique\nCapital social\nAdresse du siège\nDurée de la personne morale\nDate de clôture de l'exercice social",
        "translated": "Name or business name\nLegal form\nSocial capital\nHeadquarters address\nDuration of the legal person\nEnd date of social year",
        "bbox": (30, 235, 200, 330),
        "category": "Text",
        "lang": "fr",
    },
    {
        "text": "***Redacted***\nSociété par actions simplifiée\n6 000,00 Euros\n7 Boulevard des Alliés 94600 Choisy-le-Roi\nJusqu'au 10/04/2107\n30 septembre",
        "translated": "***Redacted***\nSimplified share company\nEUR 6,000.00\n7 Boulevard des Alliés 94600 Choisy-le-Roi\nUntil 10/04/2107\n30 September",
        "bbox": (210, 235, 480, 330),
        "category": "Text",
        "lang": "fr",
    },
    {
        "text": "**GESTION, DIRECTION, ADMINISTRATION, CONTROLE, ASSOCIES OU MEMBRES**",
        "translated": "**MANAGEMENT, DIRECTORATE, ADMINISTRATION, CONTROL, ASSOCIATES OR MEMBERS**",
        "bbox": (30, 340, 560, 355),
        "category": "Section-header",
        "lang": "fr",
    },
    {
        "text": "*Président*\nNom, prénoms\nDate et lieu de naissance\nNationalité\nDomicile personnel",
        "translated": "President\nSurname, given names\nDate and place of birth\nNationality\nPersonal home",
        "bbox": (30, 360, 200, 435),
        "category": "Text",
        "lang": "fr",
    },
    {
        "text": "GOVART Patrick Jean-Baptiste\nLe 14/09/1964 à Metz (57)\nFrançaise\n78 Rue Gabriel Péri 78420 Carrières-sur-Seine",
        "translated": "GOVART Patrick Jean-Baptiste\nThe 14/09/1964 in Metz (57)\nFrench\n78 Rue Gabriel Péri 78420 Carrières-sur-Seine",
        "bbox": (210, 370, 480, 435),
        "category": "Text",
        "lang": "fr",
    },
    {
        "text": "*Directeur général*\nNom, prénoms\nDate et lieu de naissance\nNationalité\nDomicile personnel",
        "translated": "Director-General\nSurname, given names\nDate and place of birth\nNationality\nPersonal home",
        "bbox": (30, 440, 200, 510),
        "category": "Text",
        "lang": "fr",
    },
    {
        "text": "JAMET Jean François\nLe 11/04/1963 à Paris 1st Arrondissement (75)\nFrançaise\n2 Allée du Clos de la Roseraie 94520 Périgny",
        "translated": "JAMET Jean François\nThe 11/04/1963 in Paris 1st Arrondissement (75)\nFrench\n2 Allée du Clos de la Roseraie 94520 Périgny",
        "bbox": (210, 450, 480, 510),
        "category": "Text",
        "lang": "fr",
    },
]


def main():
    # Load the test document image
    test_img_path = os.path.join("test doc", "france-certificate-of-good-standing-example .jpg")
    if not os.path.exists(test_img_path):
        print(f"Test image not found: {test_img_path}")
        print("Available files:", os.listdir("test doc"))
        return

    original = Image.open(test_img_path).convert("RGB")
    print(f"Image size: {original.size}")

    # Inpaint
    inpainted = erase_text_blocks(original, MOCK_BLOCKS)
    inpainted.save("test_inpainted.png")
    print("Saved test_inpainted.png")

    # Render
    rendered = render_translations(inpainted, original, MOCK_BLOCKS)
    rendered.save("test_rendered.png")
    print("Saved test_rendered.png")

    # Side-by-side
    w = original.width + rendered.width + 8
    h = max(original.height, rendered.height)
    combined = Image.new("RGB", (w, h), (240, 240, 240))
    combined.paste(original, (0, 0))
    combined.paste(rendered, (original.width + 8, 0))
    combined.save("test_comparison.png")
    print("Saved test_comparison.png")


if __name__ == "__main__":
    main()
