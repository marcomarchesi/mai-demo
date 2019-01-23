/*
See LICENSE folder for this sample’s licensing information.

Abstract:
View controller for selecting images and applying Vision + Core ML processing.
*/

import UIKit
import CoreML
import Vision
import ImageIO
import Photos


class ImageClassificationViewController: UIViewController {
    // MARK: - IBOutlets
    
    @IBOutlet weak var imageView: UIImageView!
    @IBOutlet weak var secondImageView: UIImageView!
    @IBOutlet weak var thirdImageView: UIImageView!
    @IBOutlet weak var fourthImageView: UIImageView!
    @IBOutlet weak var fifthImageView: UIImageView!
    
    @IBOutlet weak var firstWorstImageView: UIImageView!
    @IBOutlet weak var secondWorstImageView: UIImageView!
    @IBOutlet weak var progressView: UIProgressView!
//    @IBOutlet weak var varietyLabel: UILabel!
//    @IBOutlet weak var predictionLabel: UILabel!
    
    var images:[UIImage] = []
    var limit:Int = 36      //max number of images to be parsed
    let bestImagesLimit:Int = 7  //max number of images to be selected < limit
    let shuffledImagesLimit:Int = 4 //max number of best images to be considered for maximizing the inception score
    let varietySize:Int = 5   //number of inception score loops
    var classificationPredictions:[Float] = []
    var classificationIndex:Int = 0
    
    var scoreDict:Dictionary = [Int:Float]()
    var scoreIndex = 0
    var aestheticalScoreDict:Dictionary = [Int:Float]()
    var aestheticalScoreIndex = 0
    var technicalScoreDict:Dictionary = [Int:Float]()
    var technicalScoreIndex = 0
    var bestInceptionScore:Float = 0
    var bestIndexVariety:[Int] = []
    
    
    // setup progressView
    var counter:Int = 0 {
        didSet {
            let fractionalProgress = Float(counter) / 100.0
            let animated = counter != 0
            
            progressView.setProgress(fractionalProgress, animated: animated)
        }
    }
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        
        // fetch last N images from Photo Library
        self.fetchPhotos()
        // compute the images
        self.compute()
        
    }
    
    func fetchPhotos () {
        // Sort the images by descending creation date and fetch the first 3
        let fetchOptions = PHFetchOptions()
        fetchOptions.sortDescriptors = [NSSortDescriptor(key:"creationDate", ascending: false)]
        fetchOptions.fetchLimit = limit

        // Fetch the image assets
        let fetchResult: PHFetchResult = PHAsset.fetchAssets(with: PHAssetMediaType.image, options: fetchOptions)

        if fetchResult.count > 0 {
            for i in 0..<limit{
                fetchPhotoAtIndex(i, fetchResult)
            }
            
        }
    }
    
    // Repeatedly call the following method while incrementing
    // the index until all the photos are fetched
    func fetchPhotoAtIndex(_ index:Int, _ fetchResult: PHFetchResult<PHAsset>) {
        
        // Note that if the request is not set to synchronous
        // the requestImageForAsset will return both the image
        // and thumbnail; by setting synchronous to true it
        // will return just the thumbnail
        let requestOptions = PHImageRequestOptions()
        requestOptions.isSynchronous = true
        
        // Perform the image request
        PHImageManager.default().requestImage(for: fetchResult.object(at: index) as PHAsset, targetSize: imageView.frame.size, contentMode: PHImageContentMode.aspectFill, options: requestOptions, resultHandler: { (image, _) in
            if let image = image {
                // Add the returned image to your array
                self.images += [image]
            }
        })
    }
    
    
    // MARK: - Variety computation
    lazy var InceptionClassificationRequest: VNCoreMLRequest = {
        do {
            /*
             Use the Swift class `MobileNet` Core ML generates from the model.
             To use a different Core ML classifier model, add it to the project
             and replace `MobileNet` with that model's generated Swift class.
             */
            let model = try VNCoreMLModel(for: MobileNet().model)
            
            let request = VNCoreMLRequest(model: model, completionHandler: { [weak self] request, error in
                self?.InceptionClassification(for: request, error: error)
            })
            request.imageCropAndScaleOption = .centerCrop
            return request
        } catch {
            fatalError("Failed to load Vision ML model: \(error)")
        }
    }()
    
    
    /// - Tag: Compute Variety between selected images through Inception Score
    func InceptionClassification(for request: VNRequest, error: Error?) {
        DispatchQueue.main.sync {
            guard let results = request.results else {
//                self.varietyLabel.text = "Unable to predict.\n\(error!.localizedDescription)"
                return
            }
            // The `results` will always be `VNClassificationObservation`s, as specified by the Core ML model in this project.
            let predictions = results as! [VNClassificationObservation]
            
            if predictions.isEmpty {
//                self.varietyLabel.text = "Nothing predicted."
            } else {
                //reset classificationPredictions
                for i in 0..<(predictions.count) {
                    self.classificationPredictions.append(predictions[i].confidence)
                }
            }
        }
    }
    
    // MARK: - Image Assessment
    lazy var PredictionRequest: VNCoreMLRequest = {
        do {
            /*
             Use the Swift class `MobileNet` Core ML generates from the model.
             To use a different Core ML classifier model, add it to the project
             and replace `MobileNet` with that model's generated Swift class.
             */
            let model = try VNCoreMLModel(for: NimaAesthetic().model)
            
            let request = VNCoreMLRequest(model: model, completionHandler: { [weak self] request, error in
                self?.Prediction(for: request, error: error)
            })
            request.imageCropAndScaleOption = .centerCrop
            return request
        } catch {
            fatalError("Failed to load Vision ML model: \(error)")
        }
    }()
    
    // MARK: - Image Assessment
    lazy var TechnicalPredictionRequest: VNCoreMLRequest = {
        do {
            /*
             Use the Swift class `MobileNet` Core ML generates from the model.
             To use a different Core ML classifier model, add it to the project
             and replace `MobileNet` with that model's generated Swift class.
             */
            let model = try VNCoreMLModel(for: NimaTechnical().model)
            
            let request = VNCoreMLRequest(model: model, completionHandler: { [weak self] request, error in
                self?.TechnicalPrediction(for: request, error: error)
            })
            request.imageCropAndScaleOption = .centerCrop
            return request
        } catch {
            fatalError("Failed to load Vision ML model: \(error)")
        }
    }()
    
    /// - Tag: ProcessClassifications
    func Prediction(for request: VNRequest, error: Error?) {
        DispatchQueue.main.sync {
            guard let results = request.results else {
                return
            }
            let predictions = results as! [VNCoreMLFeatureValueObservation]
            let predictionArray = predictions[0].featureValue.multiArrayValue!
            
            scoreDict[self.scoreIndex] = calculateMeanScore(for: predictionArray)
            self.scoreIndex += 1
            
            if predictionArray.count == 0 {
            } else {
                var preds = [String]()
                for i in 0..<10 {
                    preds.append(String(format: "  (%.2f)", predictionArray[i].floatValue))
                }
            }
        }
    }
    
    func TechnicalPrediction(for request: VNRequest, error: Error?) {
        DispatchQueue.main.sync {
            guard let results = request.results else {
                return
            }
            let predictions = results as! [VNCoreMLFeatureValueObservation]
            let predictionArray = predictions[0].featureValue.multiArrayValue!
            
            technicalScoreDict[self.technicalScoreIndex] = calculateMeanScore(for: predictionArray)
            self.technicalScoreIndex += 1
            
            if predictionArray.count == 0 {
            } else {
                var preds = [String]()
                for i in 0..<10 {
                    preds.append(String(format: "  (%.2f)", predictionArray[i].floatValue))
                }
            }
        }
    }
    
    /// - Tag: Computation per image
    func compute() {
        
        // Compute predictions of aesthetical and technical scores
        DispatchQueue.global(qos: .userInitiated).async {
            
            DispatchQueue.main.async { [unowned self] in
                self.counter = 30
            }
            
            for i in 0..<self.images.count{
                let orientation = CGImagePropertyOrientation(self.images[i].imageOrientation)
                guard let ciImage = CIImage(image: self.images[i]) else { fatalError("Unable to create \(CIImage.self) from \(self.images[i]).") }

                    let handler = VNImageRequestHandler(ciImage: ciImage, orientation: orientation)
                    do {
                        try handler.perform([self.PredictionRequest]) // first scores
                        try handler.perform([self.TechnicalPredictionRequest])
                        print(self.scoreDict)
                        print(self.scoreIndex)
                    } catch {
                        print("Failed to perform classification.\n\(error.localizedDescription)")
                    }
            }
            print("Prediction DONE!")
            DispatchQueue.main.async { [unowned self] in
                self.counter = 60
            }
            // Sort results
            let sortedScoreDict  = self.scoreDict.sorted(by: { $0.value > $1.value })
            print(sortedScoreDict)
            
            // the array of indexes
            let sortedIndexes = sortedScoreDict.map { $0.key } // get the indexes of the images in order of score
            
            // get also the worst images
            var worstIndexes = [Int]()
            worstIndexes = Array(sortedIndexes.suffix(2))
//            print(worstIndexes)
        
            // pick the best image index
            let bestImageIndex = sortedIndexes[0]
            let selectedIndexes = sortedIndexes[1...self.bestImagesLimit]
            
            // Optimize Variety within best scored images
            DispatchQueue.global(qos: .userInitiated).async {
                
                // loop over a number of shuffled indexes
                for _ in 0..<self.varietySize{
                    
                    // reset the inception score
                    self.classificationPredictions.removeAll()
                    
                    //shuffle N-1 elements
                    var shuffledIndexes = Array(selectedIndexes.shuffled()[0...self.shuffledImagesLimit])
                    shuffledIndexes.append(bestImageIndex) //append the best image
                    print(shuffledIndexes)
                    
                // Calculate Inception/MobileNet Score
                    for i in shuffledIndexes{  //TODO replace with proper array of images
                        let orientation = CGImagePropertyOrientation(self.images[i].imageOrientation)
                        guard let ciImage = CIImage(image: self.images[i]) else { fatalError("Unable to create \(CIImage.self) from \(self.images[i]).") }
                        
                        let handler = VNImageRequestHandler(ciImage: ciImage, orientation: orientation)
                        do {
                            try handler.perform([self.InceptionClassificationRequest]) // then variety
                        } catch {
                            print("Failed to perform classification.\n\(error.localizedDescription)")
                        }
                    }
                    
                    DispatchQueue.main.async { [unowned self] in
                        self.counter += 8
                    }
                    
                    // Calculate Score from Predictions
                    let lastInceptionScore = inceptionScore(for: self.classificationPredictions, limit: self.limit)
                    print(lastInceptionScore)
                    if lastInceptionScore > self.bestInceptionScore {
                        self.bestIndexVariety = shuffledIndexes
                        self.bestInceptionScore = lastInceptionScore
                    }
                    
                }
                print(self.bestIndexVariety)
                print(self.bestInceptionScore)
                print("DONE!")
                
                DispatchQueue.main.async { [unowned self] in
                    self.progressView.isHidden = true
                    print("Here the best image!")
                    self.imageView.image = self.images[self.bestIndexVariety.last ?? 0]
                    self.secondImageView.image = self.images[self.bestIndexVariety[0]]
                    self.thirdImageView.image = self.images[self.bestIndexVariety[1]]
                    self.fourthImageView.image = self.images[self.bestIndexVariety[2]]
                    self.fifthImageView.image = self.images[self.bestIndexVariety[3]]
                    self.firstWorstImageView.image = self.images[worstIndexes[0]]
                    self.secondWorstImageView.image = self.images[worstIndexes[1]]
                }
            }
        }
    }
}
