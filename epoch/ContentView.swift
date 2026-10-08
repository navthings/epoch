//
//  ContentView.swift
//  epoch
//
//  Created by navneet dagdiya on 7/10/2026.
//

import SwiftUI
import Foundation

// mnist

let trainImagesPath = "train-images-idx3-ubyte"
let trainLabelsPath = "train-labels-idx1-ubyte"

func loadFile(_ path: String) -> [UInt8] {
    guard let url = Bundle.main.url(
        forResource: path,
        withExtension: nil,
        subdirectory: "data"
    ) else {
        fatalError("Could not find \(path)")
    }

    do {
        return try Array(Data(contentsOf: url))
    } catch {
        fatalError("Could not load \(path): \(error)")
    }
}

let trainImages = loadFile(trainImagesPath)
let trainLabels = loadFile(trainLabelsPath)


// mnist image

func getImage(_ index: Int) -> [Float] {
    let start = 16 + index * 784

    var pixels = [Float]()
    pixels.reserveCapacity(784)

    for i in 0..<784 {
        pixels.append(
            Float(trainImages[start + i]) / 255.0
        )
    }

    return pixels
}

func trainLabel(_ index: Int) -> Int {
    return Int(trainLabels[8 + index])
}


// math :|

func sigmoid(_ x: Float) -> Float {
    return 1.0 / (1.0 + exp(-x))
}


// layer

struct Layer {
    var weights: [[Float]]
    var biases: [Float]
}

func makeLayer(inputs: Int, neurons: Int) -> Layer {
    var weights = [[Float]]()

    for _ in 0..<neurons {
        var neuronWeights = [Float]()
        neuronWeights.reserveCapacity(inputs)

        for _ in 0..<inputs {
            neuronWeights.append(
                Float.random(in: -0.1...0.1)
            )
        }

        weights.append(neuronWeights)
    }

    return Layer(
        weights: weights,
        biases: Array(repeating: 0.0, count: neurons)
    )
}


// forward pass

func layer(
    _ inputs: [Float],
    _ layer: Layer
) -> [Float] {

    var outputs = [Float]()
    outputs.reserveCapacity(layer.weights.count)

    for j in 0..<layer.weights.count {
        var total = layer.biases[j]

        for i in 0..<inputs.count {
            total += inputs[i] * layer.weights[j][i]
        }

        outputs.append(sigmoid(total))
    }

    return outputs
}


// target

func makeTarget(_ digit: Int) -> [Float] {
    var target = Array(
        repeating: Float(0.0),
        count: 10
    )

    target[digit] = 1.0

    return target
}


// loss

func loss(
    _ output: [Float],
    _ target: [Float]
) -> Float {

    var total: Float = 0.0

    for i in 0..<output.count {
        let difference = output[i] - target[i]
        total += difference * difference
    }

    return total
}


// network

struct Network {

    var hidden: Layer
    var output: Layer

    let learningRate: Float = 0.5


    // train step

    mutating func trainStep(_ index: Int) -> Float {

        // forward pass

        let image = getImage(index)

        let hidden = layer(
            image,
            self.hidden
        )

        let output = layer(
            hidden,
            self.output
        )

        let target = makeTarget(
            trainLabel(index)
        )


        // output gradients

        var outputBlame = Array(
            repeating: Float(0.0),
            count: 10
        )

        for k in 0..<10 {

            let said = output[k]
            let wanted = target[k]

            outputBlame[k] =
                (said - wanted)
                * said
                * (1.0 - said)
        }


        // hidden gradients

        var hiddenBlame = Array(
            repeating: Float(0.0),
            count: 32
        )

        for j in 0..<32 {

            var total: Float = 0.0

            for k in 0..<10 {

                total +=
                    outputBlame[k]
                    * self.output.weights[k][j]
            }

            let said = hidden[j]

            hiddenBlame[j] =
                total
                * said
                * (1.0 - said)
        }


        // update weights output

        for k in 0..<10 {

            for j in 0..<32 {

                self.output.weights[k][j] -=
                    learningRate
                    * outputBlame[k]
                    * hidden[j]
            }

            self.output.biases[k] -=
                learningRate
                * outputBlame[k]
        }


        // update hidden weights

        for j in 0..<32 {

            for p in 0..<784 {

                self.hidden.weights[j][p] -=
                    learningRate
                    * hiddenBlame[j]
                    * image[p]
            }

            self.hidden.biases[j] -=
                learningRate
                * hiddenBlame[j]
        }


        return loss(
            output,
            target
        )
    }


    // prediction

    func predict(_ image: [Float]) -> [Float] {

        let hidden = layer(
            image,
            hidden
        )

        return layer(
            hidden,
            output
        )
    }
}


// drawing

struct Drawing {

    var strokes: [[CGPoint]] = []

    mutating func startStroke(_ point: CGPoint) {
        strokes.append([point])
    }

    mutating func addPoint(_ point: CGPoint) {

        guard !strokes.isEmpty else {
            startStroke(point)
            return
        }

        strokes[strokes.count - 1].append(point)
    }

    mutating func clear() {
        strokes.removeAll()
    }


    // convert drawing to 28 x 28

    func pixels() -> [Float] {

        var pixels = Array(
            repeating: Float(0.0),
            count: 784
        )

        for stroke in strokes {

            for point in stroke {

                let x = Int(point.x / 10.0)
                let y = Int(point.y / 10.0)

                guard
                    x >= 0,
                    x < 28,
                    y >= 0,
                    y < 28
                else {
                    continue
                }

                pixels[
                    y * 28 + x
                ] = 1.0
            }
        }

        return pixels
    }
}


// app

struct ContentView: View {

    @State private var drawing = Drawing()

    @State private var network = Network(
        hidden: makeLayer(
            inputs: 784,
            neurons: 32
        ),

        output: makeLayer(
            inputs: 32,
            neurons: 10
        )
    )

    @State private var trainingSteps = 1_000
    @State private var prediction = Array(
        repeating: Float(0.0),
        count: 10
    )

    @State private var training = false
    @State private var currentStep = 0
    @State private var currentLoss: Float = 0.0
    @State private var drawingStroke = false


    var body: some View {

        VStack(spacing: 20) {

            Text("epoch")
                .font(.largeTitle.bold())


            Canvas { context, size in

                for stroke in drawing.strokes {

                    guard let first = stroke.first else {
                        continue
                    }

                    var path = Path()

                    path.move(to: first)

                    for point in stroke.dropFirst() {
                        path.addLine(to: point)
                    }

                    context.stroke(
                        path,
                        with: .color(.black),
                        style: StrokeStyle(
                            lineWidth: 18,
                            lineCap: .round,
                            lineJoin: .round
                        )
                    )
                }
            }
            .background(.white)
            .frame(
                width: 280,
                height: 280
            )
            .clipShape(
                RoundedRectangle(
                    cornerRadius: 16
                )
            )
            .gesture(
                DragGesture(
                    minimumDistance: 0
                )
                .onChanged { value in

                    if !drawingStroke {

                        drawing.startStroke(
                            value.location
                        )

                        drawingStroke = true

                    } else {

                        drawing.addPoint(
                            value.location
                        )
                    }
                }
                .onEnded { _ in

                    drawingStroke = false

                    predict()
                }
            )


            // prediction

            VStack(spacing: 5) {

                ForEach(
                    0..<10,
                    id: \.self
                ) { digit in

                    HStack {

                        Text("\(digit)")
                            .frame(width: 20)

                        GeometryReader { geometry in

                            Rectangle()
                                .frame(
                                    width:
                                        geometry.size.width
                                        * CGFloat(
                                            prediction[digit]
                                        ),
                                    height: 10
                                )
                        }
                    }
                    .frame(height: 10)
                }
            }


            // training

            Stepper(
                "training steps: \(trainingSteps)",
                value: $trainingSteps,
                in: 100...60_000,
                step: 100
            )


            Button(
                training
                    ? "training..."
                    : "train"
            ) {
                train()
            }
            .disabled(training)


            Button("clear") {

                drawing.clear()

                prediction = Array(
                    repeating: 0.0,
                    count: 10
                )
            }


            if training {

                ProgressView(
                    value: Double(currentStep),
                    total: Double(trainingSteps)
                )
            }


            if currentLoss > 0 {

                Text(
                    "loss: \(currentLoss, specifier: "%.4f")"
                )
                .monospaced()
            }
        }
        .padding()
    }


    // prediction

    func predict() {

        guard !drawing.strokes.isEmpty else {
            return
        }

        prediction = network.predict(
            drawing.pixels()
        )
    }


    // training

    func train() {

        training = true
        currentStep = 0

        Task {

            for step in 0..<trainingSteps {

                let index = Int.random(
                    in: 0..<60_000
                )

                let currentLoss =
                    network.trainStep(index)

                if step % 10 == 0 {

                    await MainActor.run {

                        currentStep = step
                        self.currentLoss = currentLoss
                    }
                }
            }

            await MainActor.run {

                currentStep = trainingSteps
                training = false

                predict()
            }
        }
    }
}
