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
    @IBOutlet weak var varietyLabel: UILabel!
    @IBOutlet weak var predictionLabel: UILabel!
    
    var images:[UIImage] = []
    var limit:Int = 5
    var classificationPredictions:[Float] = []
    var classificationIndex:Int = 0
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        // fetch last N images from Photo Library
        self.fetchPhotos()
        // compute the images
        self.compute()
        
    }
    
    func fetchPhotos () {
        // Sort the images by descending creation date and fetch the first 3
        
//        let albumName = "Marco"
//        var assetCollection = PHAssetCollection()
//        var albumFound = Bool()
//        var photoAssets = PHFetchResult<AnyObject>()
//        let fetchOptions = PHFetchOptions()
//
//        fetchOptions.predicate = NSPredicate(format: "title = %@", albumName)
//        let collection:PHFetchResult = PHAssetCollection.fetchAssetCollections(with: .album, subtype: .any, options: fetchOptions)
        
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
                self.varietyLabel.text = "Unable to predict.\n\(error!.localizedDescription)"
                return
            }
            // The `results` will always be `VNClassificationObservation`s, as specified by the Core ML model in this project.
            let predictions = results as! [VNClassificationObservation]
            
            if predictions.isEmpty {
                self.varietyLabel.text = "Nothing predicted."
            } else {
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
    
    /// - Tag: ProcessClassifications
    func Prediction(for request: VNRequest, error: Error?) {
        DispatchQueue.main.sync {
            guard let results = request.results else {
                self.predictionLabel.text = "Unable to predict.\n\(error!.localizedDescription)"
                return
            }
            let predictions = results as! [VNCoreMLFeatureValueObservation]
            let predictionArray = predictions[0].featureValue.multiArrayValue!
            
            calculateMeanScore(for: predictionArray)
            print("Mean Score is %f", calculateMeanScore(for: predictionArray))
            
            if predictionArray.count == 0 {
                self.predictionLabel.text = "Nothing predicted."
            } else {
                var preds = [String]()
                for i in 0..<10 {
                    preds.append(String(format: "  (%.2f)", predictionArray[i].floatValue))
                }
                self.predictionLabel.text = "Prediction:\n" + preds.joined(separator: "\n")
            }
        }
    }
    
    /// - Tag: Computation per image
    func compute() {
        
        // Compute predictions of aesthetical and technical scores
        DispatchQueue.global(qos: .userInitiated).async {
            for i in 0..<self.images.count{
                let orientation = CGImagePropertyOrientation(self.images[i].imageOrientation)
                guard let ciImage = CIImage(image: self.images[i]) else { fatalError("Unable to create \(CIImage.self) from \(self.images[i]).") }

                    let handler = VNImageRequestHandler(ciImage: ciImage, orientation: orientation)
                    do {
                        try handler.perform([self.PredictionRequest]) // first scores
                    } catch {
                        print("Failed to perform classification.\n\(error.localizedDescription)")
                    }
            }
            print("Prediction DONE!")
            //                        // Sort results
            //                        // TODO
            //                        // Pick best scores
            //                        // TODO
            // Optimize Variety within best scored images
            DispatchQueue.global(qos: .userInitiated).async {
                // Calculate Inception/MobileNet Score
                for i in 0..<self.images.count{  //TODO replace with proper array of images
                    let orientation = CGImagePropertyOrientation(self.images[i].imageOrientation)
                    guard let ciImage = CIImage(image: self.images[i]) else { fatalError("Unable to create \(CIImage.self) from \(self.images[i]).") }
                    
                    let handler = VNImageRequestHandler(ciImage: ciImage, orientation: orientation)
                    do {
                        try handler.perform([self.InceptionClassificationRequest]) // then variety
                    } catch {
                        print("Failed to perform classification.\n\(error.localizedDescription)")
                    }
                }
                
                // Calculate Score from Predictions
                
                var dummyArray:[Float] = [1.0,2.0,3.0,1.0,4.5,3.0,1.0,1.5,2.0,2.5] // 10 --> (5,2)
                
                print(inceptionScore(for: dummyArray, limit: self.limit))
                print(inceptionScore(for: self.classificationPredictions, limit: self.limit))

                print("DONE!")
            }
        }
    }
}
