//
//  Utils.swift
//  Vision+ML Example
//
//  Created by Marco Marchesi on 1/19/19.
//  Copyright © 2019 Apple. All rights reserved.
//

import Foundation
import CoreML
import Vision

// Inception Score Implementation
// TODO
func predToScore(for preds: [Float], limit: Int){
    
    print(preds.count)
    
//    //let's consider (5,999)
//    var predsMean = preds           //TODO Calculate Mean over the selected images (5 by default)
//    var kl = [Float]()
//    for i in 0..<predsMean.count{
//        //            kl[i] = preds[i].confidence * (log(preds[i].confidence) - log(predsMean)
//        kl[i] = preds[i]
//    }
//    // sum of kl
//    var _sum  = calculateSum(for: kl)
//    //        var _score = calculateMean(for: )
    
}

func calculateMean(for inputs:[Float]) -> Float{
    var total:Double = 0
    for input in inputs{
        total += Double(input)
    }
    return Float(total / Double(inputs.count))
}

func calculateSum(for inputs:[Float]) -> Float{
    var total:Float = 0
    for input in inputs{
        total += input
    }
    return total
}

// calculate the mean score for the aesthetical and technical score
func calculateMeanScore(for predictionsArray:MLMultiArray) -> Double{
    
    //normalize the predictions
    var sum:Double = 0.0
    var normalizedLabels = [Double]()
    for i in 0..<10 {
        sum = sum + Double(predictionsArray[i])
    }
    // weight
    var scoreSum:Double = 0.0
    for i in 0..<10 {
        normalizedLabels.append((Double(predictionsArray[i]) / sum) * Double((i + 1)))
        scoreSum = scoreSum + normalizedLabels[i]
    }
    
    return scoreSum
}
