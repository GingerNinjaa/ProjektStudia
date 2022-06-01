using Microsoft.ML;
using System.IO;
using ProjektStudia;

namespace ProjektStudia
{
    static class Program
    {
        static string _trainDataPath = Path.Combine(Environment.CurrentDirectory, "Data", @"C:\Users\damia\Downloads\MOCK_DATAv2.csv"); //trenowanie modelu
        static string _testDataPath = Path.Combine(Environment.CurrentDirectory, "Data", @"C:\Users\damia\Downloads\MOCK_DATAv3.csv"); //Plik produkcyjny
        //static string _modelPath = Path.Combine(Environment.CurrentDirectory, "Data", "Model.zip"); //wytrenowany model
        static void Main(string[] args)
        {
            Console.WriteLine(Environment.CurrentDirectory);

            MLContext mlContext = new MLContext(seed: 0);

            //Metoda Train() wykonuje następujące zadania:
            //Ładuje dane.
            //Wyodrębnia i przekształca dane.
            //Trenuje model.
            //Zwraca model.
            var model = Train(mlContext, _trainDataPath);

            Evaluate(mlContext, model);

            //przewidywanie
            TestSinglePrediction(mlContext, model);

        }

        public static ITransformer Train(MLContext mlContext, string dataPath)
        {

            IDataView dataView = mlContext.Data.LoadFromTextFile<EntryData>(dataPath, hasHeader: true, separatorChar: ',');

            var pipeline = mlContext.Transforms.CopyColumns(outputColumnName: "Label", inputColumnName: "Ilosc")
                    //Zamieniamy Int na string
                    //.Append(mlContext.Transforms.Categorical.OneHotEncoding( "NameEncoded",  "Imie"))
                    //.Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "NazwiskoEncoded", inputColumnName: "Nazwisko"))

                    //Łączenie kilku kolumn
                    //             .Append(mlContext.Transforms.Concatenate("Features", "Id", "NameEncoded", "NazwiskoEncoded", "Nr_Pracownika", "Ilosc"))
                    .Append(mlContext.Transforms.Concatenate("Features", "Id", "Nr_Pracownika", "Ilosc"))
                    //Wybór algorytmu uczenia
                    .Append(mlContext.Regression.Trainers.FastTree());

            Console.WriteLine("=============== Create and Train the Model ===============");

            //Dopasuj model do trenowania i zwróć wytrenowany model
            var model = pipeline.Fit(dataView);

            Console.WriteLine("=============== End of training ===============");
            Console.WriteLine();
            //Zwracanie wytrenowanego modelu
            return model;
        }

        //Ocena modelu
        private static void Evaluate(MLContext mlContext, ITransformer model)
        {
            //Metoda Evaluate wykonuje następujące zadania:

            //  Ładuje testowy zestaw danych.
            //Tworzy ewaluator regresji.
            //Ocenia model i tworzy metryki.
            //Wyświetla metryki.

            //Ładowanie danych tesowych
            IDataView dataView = mlContext.Data.LoadFromTextFile<EntryData>(_testDataPath, hasHeader: true, separatorChar: ',');

            //Tworzy przewidywania
            var predictions = model.Transform(dataView);


            var metrics = mlContext.Regression.Evaluate(predictions, "Label", "Score");


            Console.WriteLine();
            Console.WriteLine($"*************************************************");
            Console.WriteLine($"*       Model quality metrics evaluation         ");
            Console.WriteLine($"*------------------------------------------------");
            //metryka oceny modeli regresjiRSquared przyjmuje wartości z wartości od 0 do 1. Im wartość jest bliższa 1, tym lepszy jest model. 
            Console.WriteLine($"*       RSquared Score:      {metrics.RSquared:0.##}");
            // jest jedną z metryk oceny modelu regresji. Tym niższa jest, tym lepszy jest model.
            Console.WriteLine($"*       Root Mean Squared Error:      {metrics.RootMeanSquaredError:#.##}");
            Console.WriteLine($"*************************************************");
        }

        //Używanie modelu do przewidywania
        private static void TestSinglePrediction(MLContext mlContext, ITransformer model)
        {
            //Metoda TestSinglePrediction wykonuje następujące zadania:

            //Tworzy pojedynczy komentarz do danych testowych.
            //Przewidywanie kwoty opłat na podstawie danych testowych.
            //Łączy dane testowe i przewidywania dotyczące raportowania.
            //Wyświetla przewidywane wyniki.

            //Prediction test
            // Create prediction function and make prediction.
            var predictionFunction = mlContext.Model.CreatePredictionEngine<EntryData, DataPrediction>(model);

            //Sample:
           //Testowy Model
            var productionSample = new EntryData()
            {
               Id = 321,
              // Imie = "Marek",
              // Nazwisko = "Podraza",
               Nr_Pracownika = 321657,
               Ilosc = 0 // To predict
            };
          
            var prediction = predictionFunction.Predict(productionSample);
            Console.WriteLine($"**********************************************************************");
            Console.WriteLine($"Predicted fare: {prediction.PrzewidywanaIlość:0.####}, średnia: 56");
            Console.WriteLine($"**********************************************************************");

        }

    }


}

