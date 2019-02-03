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
import QuartzCore


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
    @IBOutlet weak var progressLabel: UILabel!
//    @IBOutlet weak var varietyLabel: UILabel!
//    @IBOutlet weak var predictionLabel: UILabel!
    
    @IBOutlet weak var timeLabel: UILabel!
    
    var aFirstImageLabel:UILabel!
    var tFirstImageLabel:UILabel!
    var aSecondImageLabel:UILabel!
    var tSecondImageLabel:UILabel!
    var aThirdImageLabel:UILabel!
    var tThirdImageLabel:UILabel!
    var aFourthImageLabel:UILabel!
    var tFourthImageLabel:UILabel!
    var aFifthImageLabel:UILabel!
    var tFifthImageLabel:UILabel!
    var aFirstWorstImageLabel:UILabel!
    var tFirstWorstImageLabel:UILabel!
    var aSecondWorstImageLabel:UILabel!
    var tSecondWorstImageLabel:UILabel!
    
    var images:[UIImage] = []
    var limit:Int = 500      //max number of images to be parsed
    let bestImagesLimit:Int = 7  //max index among the best images to be selected < limit
    let shuffledImagesLimit:Int = 4 //max index among the best images to be considered for calculating the inception score
    let varietySize:Int = 5   //number of inception score loops
    var tW = 0.5
    var aW = 0.5
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
    var worstIndexes = [Int]()
    
    var start:Double = 0
    
    var startDate:NSDate? = nil
    var stopDate:NSDate? = nil
    
    // setup progressView
    var counter:Int = 0 {
        didSet {
            let fractionalProgress = Float(counter) / 100.0
            let animated = counter != 0
            
            progressView.setProgress(fractionalProgress, animated: animated)
        }
    }
    
    enum scoreType {
        case AESTHETIC
        case TECHNICAL
    }
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        self.timeLabel.isHidden = true
        start = CACurrentMediaTime()
        self.progressLabel.text = "Loading photos..."
        self.progressLabel.isHidden = true
        self.progressView.isHidden = true
        
        
        self.aFirstImageLabel = UILabel(frame: CGRect(x:0,y:0, width:50, height: 20))
        self.tFirstImageLabel = UILabel(frame: CGRect(x:0,y:0, width:50, height: 20))
        self.aSecondImageLabel = UILabel(frame: CGRect(x:0,y:0, width:50, height: 20))
        self.tSecondImageLabel = UILabel(frame: CGRect(x:0,y:0, width:50, height: 20))
        self.aThirdImageLabel = UILabel(frame: CGRect(x:0,y:0, width:50, height: 20))
        self.tThirdImageLabel = UILabel(frame: CGRect(x:0,y:0, width:50, height: 20))
        self.aFourthImageLabel = UILabel(frame: CGRect(x:0,y:0, width:50, height: 20))
        self.tFourthImageLabel = UILabel(frame: CGRect(x:0,y:0, width:50, height: 20))
        self.aFifthImageLabel = UILabel(frame: CGRect(x:0,y:0, width:50, height: 20))
        self.tFifthImageLabel = UILabel(frame: CGRect(x:0,y:0, width:50, height: 20))
        self.aFirstWorstImageLabel = UILabel(frame: CGRect(x:0,y:0, width:50, height: 20))
        self.tFirstWorstImageLabel = UILabel(frame: CGRect(x:0,y:0, width:50, height: 20))
        self.aSecondWorstImageLabel = UILabel(frame: CGRect(x:0,y:0, width:50, height: 20))
        self.tSecondWorstImageLabel = UILabel(frame: CGRect(x:0,y:0, width:50, height: 20))
        
        
    
        // fetch last N images from Photo Library
//        self.fetchPhotos()
        
//        let formatter = DateFormatter()
//        formatter.dateFormat = "MM-dd-yyyy"
//        self.fetchPhotosInRange(startDate: formatter.date(from: "04-06-2015")! as NSDate, endDate: formatter.date(from: "04-16-2019")! as NSDate)
//        self.fetchPhotosInRange(startDate: self.startDate!, endDate: self.stopDate!)
//        // compute the images
//        if self.images.count > 10{
//            self.progressLabel.isHidden = false
//            self.progressView.isHidden = false
//            self.compute()
//        }
//        else{
//            print("No images")
//            let alert = UIAlertController(title: "H4 with Guide Pins", message: "Can it be removed?", preferredStyle: UIAlertControllerStyle.alert)
//            alert.addAction(UIAlertAction(title:"OK", style: .default, handler:  { action in self.performSegue(withIdentifier: "FromResultsToSelection", sender: self) }))
//            self.present(alert, animated: true, completion: nil)
//        }
        
    }
    
    override func viewDidAppear(_ animated: Bool) {
        super.viewDidAppear(animated)
        self.fetchPhotosInRange(startDate: self.startDate!, endDate: self.stopDate!)
        // compute the images
        if self.images.count > 10{
            self.progressLabel.isHidden = false
            self.progressView.isHidden = false
            self.compute()
        }
        else{
            let alert = UIAlertController(title: "Error", message: "Select a time interval that contains more than 10 images", preferredStyle: UIAlertControllerStyle.alert)
            alert.addAction(UIAlertAction(title:"OK", style: .default, handler:  { action in self.performSegue(withIdentifier: "FromResultsToSelection", sender: self) }))
            self.present(alert, animated: true, completion: nil)
        }
        
    }
    
    func makeLabel(label:UILabel, vw:UIImageView, y:CGFloat) {
//        label = UILabel(frame: CGRect(x:0,y:0, width:200, height: 50))
        view.addSubview(label)
        label.textAlignment = .center
        label.backgroundColor = UIColor.black
        label.adjustsFontSizeToFitWidth = true
        label.frame.origin.x = vw.frame.origin.x + vw.frame.width - 50
        label.frame.origin.y = vw.frame.origin.y + vw.frame.height - 40 + y
        label.textColor =  UIColor.white
        label.font = UIFont(name: "DIN-Alternate-Bold", size: 20)
        
    }
    
    func makeText(label:UILabel, value:Float!, type:scoreType){
        var labelValue:Float! = value
        labelValue = labelValue.rounded(toPlaces: 2)
        if type == scoreType.TECHNICAL {
            label.text = "t" + String(labelValue)
        } else
        {
            label.text = "a" + String(labelValue)
        }
    }
    
    func fetchPhotosInRange(startDate:NSDate, endDate:NSDate) {
        
        let requestOptions = PHImageRequestOptions()
        requestOptions.isSynchronous = true
        requestOptions.isNetworkAccessAllowed = true
        
        // Fetch the images between the start and end date
        let fetchOptions = PHFetchOptions()
        fetchOptions.predicate = NSPredicate(format: "creationDate > %@ AND creationDate < %@", startDate, endDate)
        
        self.counter = 0
        
        let fetchResult: PHFetchResult = PHAsset.fetchAssets(with: PHAssetMediaType.image, options: fetchOptions)
        if fetchResult.count > 0 {
            for i in 0..<fetchResult.count{
                self.counter += 30 / fetchResult.count
                fetchPhotoAtIndex(i, fetchResult)
            }
        }
    }
    
    func fetchPhotos () {
        // Sort the images by descending creation date and fetch the first 3
        let fetchOptions = PHFetchOptions()
        fetchOptions.sortDescriptors = [NSSortDescriptor(key:"creationDate", ascending: false)]
        fetchOptions.fetchLimit = limit
        
        self.counter = 0

        // Fetch the image assets
        let fetchResult: PHFetchResult = PHAsset.fetchAssets(with: PHAssetMediaType.image, options: fetchOptions)

        print(fetchResult.count)
        
        if fetchResult.count > 0 {
            for i in 0..<limit{
                self.counter += 30 / fetchResult.count
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
                return
            }
            // The `results` will always be `VNClassificationObservation`s, as specified by the Core ML model in this project.
            let predictions = results as! [VNClassificationObservation]
            
            if predictions.isEmpty {
            } else {
                //reset classificationPredictions
                for i in 0..<(predictions.count) {
                    self.classificationPredictions.append(predictions[i].confidence)
                }
            }
        }
    }
    
    // MARK: - Image Assessment
    lazy var AestheticalPredictionRequest: VNCoreMLRequest = {
        do {
            /*
             Use the Swift class `MobileNet` Core ML generates from the model.
             To use a different Core ML classifier model, add it to the project
             and replace `MobileNet` with that model's generated Swift class.
             */
            let model = try VNCoreMLModel(for: NimaAesthetic().model)
            
            let request = VNCoreMLRequest(model: model, completionHandler: { [weak self] request, error in
                self?.AestheticalPrediction(for: request, error: error)
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
    func AestheticalPrediction(for request: VNRequest, error: Error?) {
        DispatchQueue.main.sync {
            guard let results = request.results else {
                return
            }
            let predictions = results as! [VNCoreMLFeatureValueObservation]
            let predictionArray = predictions[0].featureValue.multiArrayValue!
            
            aestheticalScoreDict[self.aestheticalScoreIndex] = calculateMeanScore(for: predictionArray)
            self.aestheticalScoreIndex += 1
            
//            print(aestheticalScoreDict)
            
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
                self.progressLabel.text = "Calculating " + String(self.images.count) + " scores..."
                self.counter = 30
            }
            
            for i in 0..<self.images.count{
                let orientation = CGImagePropertyOrientation(self.images[i].imageOrientation)
                guard let ciImage = CIImage(image: self.images[i]) else { fatalError("Unable to create \(CIImage.self) from \(self.images[i]).") }

                    let handler = VNImageRequestHandler(ciImage: ciImage, orientation: orientation)
                    do {
                        try handler.perform([self.AestheticalPredictionRequest]) // first scores
                        try handler.perform([self.TechnicalPredictionRequest])
                    } catch {
                        print("Failed to perform classification.\n\(error.localizedDescription)")
                    }
            }
            print("Prediction DONE!")
            DispatchQueue.main.async { [unowned self] in
                self.counter = 60
            }
            
//            print(self.technicalScoreDict)
//            print(self.aestheticalScoreDict)
            
            let maxTechnicalScore = self.technicalScoreDict.values.max()
            let maxAestheticalScore = self.aestheticalScoreDict.values.max()
            
//            print(maxTechnicalScore ?? 0)
//            print(maxAestheticalScore ?? 0)
            
            for i in 0..<self.images.count{
                self.scoreDict[i] = Float(self.tW) / Float(maxTechnicalScore ?? 0) * Float(self.technicalScoreDict[i] ?? 0) + Float(self.aW) / Float(maxAestheticalScore ?? 0) * Float( self.aestheticalScoreDict[i] ?? 0)
//                self.scoreDict[i] =  Float(self.technicalScoreDict[i] ?? 0) + Float( self.aestheticalScoreDict[i] ?? 0)
            }
            
//            print(self.scoreDict)
            
            // Sort results
            let sortedScoreDict  = self.scoreDict.sorted(by: { $0.value > $1.value })
//            print(sortedScoreDict)
            
            // the array of indexes
            let sortedIndexes = sortedScoreDict.map { $0.key } // get the indexes of the images in order of score
            
            // get also the worst images
            
            self.worstIndexes = Array(sortedIndexes.suffix(2))

        
            // pick the best image index
            let bestImageIndex = sortedIndexes[0]
            let selectedIndexes = sortedIndexes[1...self.bestImagesLimit]
            
            // Optimize Variety within best scored images
            DispatchQueue.global(qos: .userInitiated).async {
                
                // loop over a number of shuffled indexes
                for _ in 0..<self.varietySize{
                    
                    // reset the inception score
                    self.classificationPredictions.removeAll()
                    
                    //shuffle N-1 elements  (RANDOM OPTIMIZATION)
                    var shuffledIndexes = Array(selectedIndexes.shuffled()[0...self.shuffledImagesLimit])
                    shuffledIndexes.append(bestImageIndex) //append the best image
                    print(shuffledIndexes)
                    
                    // TODO: HILL CLIMBING
                    
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
                        self.progressLabel.text = "Optimizing variety..."
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
//                print("Best variety: ", self.bestIndexVariety)
//                print(self.bestInceptionScore)
//                print("DONE!")
                
                let end = CACurrentMediaTime()
                
                let timeString = "Processed in " + String(roundToNearestQuarter(num: Float(end - self.start))) + " seconds"
                print(timeString)
                
                DispatchQueue.main.async { [unowned self] in
                    
                    print(self.images.count)
                    self.progressView.isHidden = true
                    self.progressLabel.isHidden = true
                    print("Here the best image!")
                    self.timeLabel.text = timeString
                    self.timeLabel.isHidden = false
                    self.imageView.image = self.images[self.bestIndexVariety.last ?? 0]
                    self.secondImageView.image = self.images[self.bestIndexVariety[0]]
                    self.thirdImageView.image = self.images[self.bestIndexVariety[1]]
                    self.fourthImageView.image = self.images[self.bestIndexVariety[2]]
                    self.fifthImageView.image = self.images[self.bestIndexVariety[3]]
                    self.firstWorstImageView.image = self.images[self.worstIndexes[0]]
                    self.secondWorstImageView.image = self.images[self.worstIndexes[1]]
                    
                    
                    // TODO add labels for A and T scores
                    self.makeLabel(label: self.aFirstImageLabel, vw: self.imageView, y:0)
                    self.makeText(label: self.aFirstImageLabel, value: self.aestheticalScoreDict[self.bestIndexVariety.last!], type: scoreType.AESTHETIC)
                    self.makeLabel(label: self.tFirstImageLabel, vw: self.imageView, y: 20)
                    self.makeText(label: self.tFirstImageLabel, value: self.technicalScoreDict[self.bestIndexVariety.last!], type: scoreType.TECHNICAL)
                    
                    self.makeLabel(label: self.aSecondImageLabel, vw: self.secondImageView, y:0)
                    self.makeText(label: self.aSecondImageLabel, value: self.aestheticalScoreDict[self.bestIndexVariety[0]], type: scoreType.AESTHETIC)
                    self.makeLabel(label: self.tSecondImageLabel, vw: self.secondImageView, y: 20)
                    self.makeText(label: self.tSecondImageLabel, value: self.technicalScoreDict[self.bestIndexVariety[0]], type: scoreType.TECHNICAL)
                    
                    self.makeLabel(label: self.aThirdImageLabel, vw: self.thirdImageView, y:0)
                    self.makeText(label: self.aThirdImageLabel, value: self.aestheticalScoreDict[self.bestIndexVariety[1]], type: scoreType.AESTHETIC)
                    self.makeLabel(label: self.tThirdImageLabel, vw: self.thirdImageView, y: 20)
                    self.makeText(label: self.tThirdImageLabel, value: self.technicalScoreDict[self.bestIndexVariety[1]], type: scoreType.TECHNICAL)
                    
                    self.makeLabel(label: self.aFourthImageLabel, vw: self.fourthImageView, y:0)
                    self.makeText(label: self.aFourthImageLabel, value: self.aestheticalScoreDict[self.bestIndexVariety[2]], type: scoreType.AESTHETIC)
                    self.makeLabel(label: self.tFourthImageLabel, vw: self.fourthImageView, y: 20)
                    self.makeText(label: self.tFourthImageLabel, value: self.technicalScoreDict[self.bestIndexVariety[2]], type: scoreType.TECHNICAL)
                    
                    self.makeLabel(label: self.aFifthImageLabel, vw: self.fifthImageView, y:0)
                    self.makeText(label: self.aFifthImageLabel, value: self.aestheticalScoreDict[self.bestIndexVariety[3]], type: scoreType.AESTHETIC)
                    self.makeLabel(label: self.tFifthImageLabel, vw: self.fifthImageView, y: 20)
                    self.makeText(label: self.tFifthImageLabel, value: self.technicalScoreDict[self.bestIndexVariety[3]], type: scoreType.TECHNICAL)
                    
                    self.makeLabel(label: self.aFirstWorstImageLabel, vw: self.firstWorstImageView, y:0)
                    self.makeText(label: self.aFirstWorstImageLabel, value: self.aestheticalScoreDict[self.worstIndexes[0]], type: scoreType.AESTHETIC)
                    self.makeLabel(label: self.tFirstWorstImageLabel, vw: self.firstWorstImageView, y: 20)
                    self.makeText(label: self.tFirstWorstImageLabel, value: self.technicalScoreDict[self.worstIndexes[0]], type: scoreType.TECHNICAL)

                    self.makeLabel(label: self.aSecondWorstImageLabel, vw: self.secondWorstImageView, y:0)
                    self.makeText(label: self.aSecondWorstImageLabel, value: self.aestheticalScoreDict[self.worstIndexes[1]], type: scoreType.AESTHETIC)
                    self.makeLabel(label: self.tSecondWorstImageLabel, vw: self.secondWorstImageView, y: 20)
                    self.makeText(label: self.tSecondWorstImageLabel, value: self.technicalScoreDict[self.worstIndexes[1]], type: scoreType.TECHNICAL)

                    
                    
                }
            }
        }
    }
}

extension Float {
    /// Rounds the double to decimal places value
    func rounded(toPlaces places:Int) -> Float {
        let divisor = pow(10.0, Float(places))
        return (self * divisor).rounded() / divisor
    }
}
