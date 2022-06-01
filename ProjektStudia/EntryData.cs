using Microsoft.ML.Data;

namespace ProjektStudia
{
    public class EntryData
    {
        [LoadColumn(0)]
        public float Id;
        //[LoadColumn(1)]
        //public string Imie;
        //[LoadColumn(2)]
        //public string Nazwisko;
        [LoadColumn(1)]
        public float Nr_Pracownika;
        [LoadColumn(2)]
        public float Ilosc;
 
    }

    public class DataPrediction
    {
        [ColumnName("Score")]
        public float PrzewidywanaIlość;
    }
}
