"""
test_renderer.py — Test the renderer with mock OCR blocks against the test document.

Simulates what dots.mocr would return for the French certificate document,
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

# ── Mock OCR blocks (simulating dots.mocr output for the French certificate) ──
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
        "translated": "MAIN IMMATRICULATION EXTRACT IN THE TRADE AND SOCIETY REGISTER\nas of 11 September 2024",
        "bbox": (70, 120, 560, 155),
        "category": "Title",
        "lang": "fr",
    },
    {
        "text": "**IDENTIFICATION DE LA PERSONNE MORALE**",
        "translated": "IDENTIFICATION OF THE MORAL PERSON",
        "bbox": (30, 160, 560, 175),
        "category": "Section-header",
        "lang": "fr",
    },
    # ── Form block: labels and values as alternating lines (real OCR pattern) ──
    {
        "text": "Immatriculation au RCS, numéro\n***Redacted***\nDate d'immatriculation\n10/04/2008\nImmatriculation radiée le\n27/06/2024",
        "translated": "Registration at SCN, number\n***Redacted***\nDate of registration\n10/04/2008\nRegistration deleted on\n27/06/2024",
        "bbox": (30, 180, 480, 240),
        "category": "Text",
        "lang": "fr",
    },
    {
        "text": "Dénomination ou raison sociale\n***Redacted***\nForme juridique\nSociété par actions simplifiée\nCapital social\n6 000,00 Euros\nAdresse du siège\n7 Boulevard des Alliés 94600 Choisy-le-Roi\nDurée de la personne morale\nJusqu'au 10/04/2107\nDate de clôture de l'exercice social\n30 septembre",
        "translated": "Name or business name\n***Redacted***\nLegal form\nSimplified share company\nSocial capital\nEUR 6,000.00\nHeadquarters address\n7 Boulevard des Alliés 94600 Choisy-le-Roi\nDuration of the legal person\nUntil 10/04/2107\nEnd date of social year\n30 September",
        "bbox": (30, 245, 480, 380),
        "category": "Text",
        "lang": "fr",
    },
    {
        "text": "**GESTION, DIRECTION, ADMINISTRATION, CONTROLE, ASSOCIES OU MEMBRES**",
        "translated": "MANAGEMENT, DIRECTORATE, ADMINISTRATION, CONTROL, ASSOCIATES OR MEMBERS",
        "bbox": (30, 385, 560, 400),
        "category": "Section-header",
        "lang": "fr",
    },
    {
        "text": "*Président*\nNom, prénoms\nGOVART Patrick Jean-Baptiste\nDate et lieu de naissance\nLe 14/09/1964 à Metz (57)\nNationalité\nFrançaise\nDomicile personnel\n78 Rue Gabriel Péri 78420 Carrières-sur-Seine",
        "translated": "President\nSurname, given names\nGOVART Patrick Jean-Baptiste\nDate and place of birth\nThe 14/09/1964 in Metz (57)\nNationality\nFrench\nPersonal home\n78 Rue Gabriel Péri 78420 Carrières-sur-Seine",
        "bbox": (30, 405, 480, 495),
        "category": "Text",
        "lang": "fr",
    },
    {
        "text": "*Directeur général*\nNom, prénoms\nJAMET Jean François\nDate et lieu de naissance\nLe 11/04/1963 à Paris 1st Arrondissement (75)\nNationalité\nFrançaise\nDomicile personnel\n2 Allée du Clos de la Roseraie 94520 Périgny",
        "translated": "Director-General\nSurname, given names\nJAMET Jean François\nDate and place of birth\nThe 11/04/1963 in Paris 1st Arrondissement (75)\nNationality\nFrench\nPersonal home\n2 Allée du Clos de la Roseraie 94520 Périgny",
        "bbox": (30, 500, 480, 580),
        "category": "Text",
        "lang": "fr",
    },
    # ── "RENSEIGNEMENTS" section WITH standalone note line (13 lines, odd) ──
    {
        "text": "**RENSEIGNEMENTS RELATIFS A L'ACTIVITE ET A L'ETABLISSEMENT PRINCIPAL**",
        "translated": "INFORMATION CONCERNING MAIN ACTIVITIES AND ESTABLISHMENT",
        "bbox": (30, 585, 560, 600),
        "category": "Section-header",
        "lang": "fr",
    },
    {
        "text": "Adresse de l'établissement\n7 Boulevard des Alliés 94600 Choisy-le-Roi\nNom commercial\n***Redacted***\nActivité(s) exercée(s)\nPratique de l'exercice vétérinaire\nDate de commencement d'activité\n01/04/2008\nEn attente de la production de la pièce justifiant de la capacité\nOrigine du fonds ou de l'activité\nCréation\nMode d'exploitation\nExploitation directe",
        "translated": "Address of establishment\n7 Boulevard des Allied 94600 Choisy-le-Roi\nTrade name\nRedacted\nActivity(s)\nVeterinary practice\nStart date\n01/04/2008\nPending the production of the part supporting the capacity\nOrigin of fund or activity\nEstablishment\nOperating method\nDirect exploitation",
        "bbox": (30, 605, 480, 710),
        "category": "Text",
        "lang": "fr",
    },
    # ── Second "RENSEIGNEMENTS" section (12 lines, even) ──
    {
        "text": "**RENSEIGNEMENTS RELATIFS AUX AUTRES ETABLISSEMENTS DANS LE RESSORT**",
        "translated": "INFORMATION RELATING TO OTHER ESTABLISHMENTS IN THE RESENT",
        "bbox": (30, 715, 560, 730),
        "category": "Section-header",
        "lang": "fr",
    },
    {
        "text": "Adresse de l'établissement\n47 Rue de Paris 94470 Boissy-Saint-Léger\nActivité(s) exercée(s)\nClinique vétérinaire\nDate de commencement d'activité\n31/01/2011\nOrigine du fonds ou de l'activité\nCréation\nMode d'exploitation\nExploitation directe\nAdresse de l'établissement\nCentre Commercial Village 3 Rue de la Résistance 94320 Thiais",
        "translated": "Address of establishment\n47 Rue de Paris 94470 Boissy-Saint-Léger\nActivity(s)\nVeterinary clinic\nStart date\n31/01/2011\nOrigin of fund or activity\nEstablishment\nOperating method\nDirect exploitation\nAddress of establishment\nCentre Commercial Village 3 Rue de la Résistance 94320 Thiais",
        "bbox": (30, 735, 480, 850),
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
