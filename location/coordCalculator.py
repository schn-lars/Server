import csv
import pandas as pd

class CoordCalculator:

    def __init__(self):
        pass

    def calculateCoord(self, coordE, coordN):
        y_s = (coordE - 2600000) / 1000000
        x_s = (coordN - 1200000) / 1000000
        lambda_s = 2.6779094 + (4.728982 * y_s) + (0.791484 * y_s * x_s) + (0.1306 * y_s * x_s * x_s) - (0.0436 * y_s * y_s * y_s)
        sigma_s = 16.9023892 + (3.238272 * x_s) - (0.270978 * y_s * y_s) - (0.002528 * x_s * x_s) - (0.0447 * y_s * y_s * x_s) - (0.0140 * x_s * x_s * x_s)
        lamb = lambda_s * 100 / 36
        sig = sigma_s * 100 / 36
        #print(y_s, ", ", x_s)
        #print(lambda_s, ",", sigma_s)
        #print(lamb, ",", sig)
        return sig, lamb

    def handleCSV(self, writer, path_to_csv):
        df = pd.read_csv(path_to_csv, sep=";")
        #df.rename(
	#    columns={"Taxon ID": "ArtNumber", "Gruppe": "ClassName", "Trivialname": "TrivialName", "Taxon": "SpeciesName", "Link zum Taxon": "Link",
        #             "Gemeinde(n)": "City", "Kanton(e) ": "Kanton", "Fundort": "Locality", "Höhe": "Elevation", "Koordx": "LVCoordX", "Koordy": "LVCoordY"
        #            ,"Radius": "UncertaintyInMeters", "Räumliche Zusammenfassung": "RaumlicheZusammenfassung", "n Nachweise ": "nNachweis", "Jahr": "YearName",
        #             "Rote Liste": "RoteListe", "Priorität CH": "PrioCH"},
        #    inplace=True)
        for row in df.itertuples():
            CoordX, CoordY = self.calculateCoord(row.ADR_EASTING, row.ADR_NORTHING)
            #prio = row.PrioCH if pd.notna(row.PrioCH) else ""
            #uzl = row.UZL if pd.notna(row.UZL) else ""

            writer.writerow([row.ADR_EGAID, row.STR_ESID, row.BDG_EGID, row.ADR_EDID, row.STN_LABEL, row.ADR_NUMBER, row.BDG_CATEGORY, row.BDG_NAME, 
            row.ZIP_LABEL, row.COM_FOSNR, row.COM_NAME,
            row.COM_CANTON, row.ADR_STATUS, row.ADR_OFFICIAL, row.ADR_MODIFIED, CoordX, CoordY])


    def main(self):

        with open('./adresses_new.csv', mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file, delimiter='|')
            writer.writerow(["ADR_EGAID", "STR_ESID", "BDG_EGID","ADR_EDID","STN_LABEL", "ADR_NUMBER", "BDG_CATEGORY", "BDG_NAME", "ZIP_LABEL",
            "COM_FOSNR", "COM_NAME", "COM_CANTON",
            "ADR_STATUS", "ADR_OFFICIAL", "ADR_MODIFIED", "COORDX", "COORDY"])
            self.handleCSV(writer, "./adresses.csv")
            #self.handleCSV(writer, "../vw_AI.csv")
            #self.handleCSV(writer, "../vw_AR.csv")
            #self.handleCSV(writer, "../vw_BE.csv")
            #self.handleCSV(writer, "../vw_BL.csv")
            #self.handleCSV(writer, "../vw_BS.csv")
            #self.handleCSV(writer, "../vw_GE.csv")
            #self.handleCSV(writer, "../vw_SG.csv")
            #self.handleCSV(writer, "../vw_SO.csv")
            #self.handleCSV(writer, "../vw_VD.csv")
            #self.handleCSV(writer, "../vw_VS.csv")
            #self.handleCSV(writer, "../vw_ZG.csv")
            #self.handleCSV(writer, "../vw_ZH.csv")

if __name__ == "__main__":
    CoordCalculator().main()
