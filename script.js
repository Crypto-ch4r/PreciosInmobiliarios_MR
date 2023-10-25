const getData = async () => {
    const datosCasaR = await fetch("datos.json");
    const datosCasas = await datosCasaR.json();
    let datosLimpios = datosCasas.map(casa => (
        {
            precio: casa.Precio,
            cuartos: casa.NumeroDeCuartosPromedio
        }
    ));
    datosLimpios = datosLimpios.filter(casa =>
        casa.precio != null && casa.cuartos != null
    );

    return datosLimpios;
}

async function cargarModelo(){
    const uploadJSONinput = document.getElementById('upload-json');
    const uploadWeightsInput = document.getElementById('upload-weights');

    modelo = await tf.loadLayersModel(tf.io.browserFiles(
        [uploadJSONinput.files[0],uploadWeightsInput.files[0]]));
        console.log("Modelo cargado")
}

async function verCurvaInferencia(){
    var data = await getData();
    var tensorData = await convertirDatosATensores(data);
    const {entradasMax, entradasMin, etiquetasMin, etiquetasMax} = tensorData

    const [xs, preds] = tf.tidy(()=>{
        const xs = tf.linspace(0,1,100);
        const preds = modelo.predict(xs.reshape([100,1]));
        const desnormX = xs.mul(entradasMax.sub(entradasMin)).add(entradasMin);
        const desnormY = preds.mul(entradasMax.sub(entradasMin)).add(etiquetasMin);

        return [desnormX.dataSync(), desnormY.dataSync()];
    });

    const puntosPrediccion = Array.from(xs).map((val,i)=>{
        return {x:val,y:preds[i]}
    });

    const puntosOriginales = datos.map(d=>({
        x: d.cuartos, y: d.precio,
    }));


    tfvis.render.scatterplot({names: 'Predicciones vs Orgininales'},
    {values:[puntosOriginales, puntosPrediccion], series:['originales','prediccion']},
    {
        xLabel: 'Cuartos',
        yLabel: 'Precios',
        height:300
    });
}


const visualizarDatos = data => {
    const valores = data.map(d => ({
        x: d.cuartos,
        y: d.precio
    }));
    tfvis.render.scatterplot(
        { name: 'Cuartos vs Precio' },
        { values: valores },
        {
            xLabel: 'Cuartos',
            yLabel: 'Precio',
            height: 300
        }
    );
};

const crearModelo = () => {
    const modelo = tf.sequential();

    modelo.add(tf.layers.dense({ inputShape: [1], units: 1, useBias: true }));
    modelo.add(tf.layers.dense({ units: 1, useBias: true }));

    return modelo;
};

const convertirDatosATensores = data => {
    return tf.tidy(() => {
        tf.util.shuffle(data);

        const entradas = data.map(d => d.cuartos)
        const etiquetas = data.map(d => d.precio);

        const tensorEntradas = tf.tensor2d(entradas, [entradas.length, 1]);
        const tensorEtiquetas = tf.tensor2d(etiquetas, [etiquetas.length, 1]);

        const entradasMax = tensorEntradas.max();
        const entradasMin = tensorEntradas.min();
        const etiquetasMax = tensorEtiquetas.max();
        const etiquetasMin = tensorEtiquetas.min();

        const entradasNormalizadas = tensorEntradas.sub(entradasMin).div(entradasMax.sub(entradasMin));
        const etiquetasNormalizadas = tensorEtiquetas.sub(etiquetasMin).div(etiquetasMax.sub(etiquetasMin));

        return {
            entradas: entradasNormalizadas,
            etiquetas: etiquetasNormalizadas,
            entradasMax,
            entradasMin,
            etiquetasMax,
            etiquetasMin,
        }
    });
};

const entrenarModelo = async (model, inputs, labels) => {
    model.compile({
        optimizer: tf.train.adam(),
        loss: tf.losses.meanSquaredError,
        metrics: ['mse']
    });

    const surface = { name: 'Muestra historial', tab: 'Entrenamiento' };
    const tamanoBatch = 28;
    const epochs = 50;
    const history = [];

    var stopTraining;


    return await model.fit(inputs, labels, {
        tamanoBatch,
        epochs,
        shuffle: true,
        callbacks:{
            onEpochEnd: (epoch, log)=>{
                history.push(log);
                tfvis.show.history(surface, history,['loss', 'mse']);
                if(stopTraining){
                    modelo.stopTraining = true
                }
            }

        }
        
        /*tfvis.show.fitCallbacks(
            {name: 'Training Performance'},
            ['loss', 'mse'],
            { height:200, callbacks: ['onEpochEnd'] }
        )*/
    });
};

async function guardarModelo(){
    const saveResult = await modelo.save('Downloads://modelo_regresion');
}

const run = async () => {
    const data = await getData();
    visualizarDatos(data);
    const modelo = crearModelo();

    const tensorData = convertirDatosATensores(data);
    const { entradas, etiquetas } = tensorData;
    entrenarModelo(modelo, entradas, etiquetas);
};



document.addEventListener('DOMContentLoaded', run);