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
func inceptionScore(for preds: [Float], limit: Int) -> Float{
    
    let categories: Int = preds.count / limit
    var meanPreds = [Float]()
    
    for i in 0..<categories{
        var total:Float = 0
        for k in 0..<limit{
            total += preds[i + k * categories]
        }
        total = total / Float(limit)
        meanPreds.append(total)
    }
    
    var logPreds = [Float]()
    var logMeanPreds = [Float]()
    var index = 0
    for i in 0..<preds.count{
        logPreds.append(log(preds[i]))
    }
    
    for i in 0..<categories{
        logMeanPreds.append(log(meanPreds[i]))
    }
    
//    print(logPreds)
//    print(logMeanPreds)
    
    // calculate KL Divergence
    var kl = [Float]()
    for i in 0..<preds.count{
        kl.append(preds[i] * (logPreds[i] - logMeanPreds[i % categories]))
    }
//    print(kl)
    
    var klSums = [Float]()
    // mean of sum of kl
    index = 0
    var total:Float = 0.0
    for i in 0..<preds.count{
        total = total + kl[i]
        index += 1
        if index == categories{
            index = 0
            klSums.append(total)
            total = 0.0
        }
    }
//    print(klSums)
    var klMean = calculateMean(for: klSums)
//    print(klMean)
    
    return exp(klMean)
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
func calculateMeanScore(for predictionsArray:MLMultiArray) -> Float{
    
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
    
    return Float(scoreSum)
}

func roundToNearestQuarter(num : Float) -> Float {
    return round(num * 4.0)/4.0
}
