let wrapper = document.getElementById('wrapper');
let canvas = document.getElementById('drawing_board');
let eraser = document.getElementById('eraser');
let pencil = document.getElementById('pencil');
let select = document.getElementById('select');
let trash = document.getElementById('trash');
let undo = document.getElementById('undo');
let redo = document.getElementById('redo');
let collapser = document.getElementById('collapser');

let submit = document.getElementById('guess');
let loading = document.getElementById('loading');

let drawer = canvas.getContext('2d');
let body = document.body;

let active = false;
let drawing = false;
let tool = 'select';

let prev_x = 0;
let prev_y = 0;
let length = 0;

const mainSize = 477;
const mainEraser = 40;
const mainPencil = 20;

const auxSize = 50;
const auxPencil = 3;
const auxEraser = 5;
const scalingFactor = auxSize/mainSize;

let smallCanvas = document.createElement('canvas');
smallCanvas.setAttribute('width', auxSize + 'px');
smallCanvas.setAttribute('height', auxSize + 'px');
smallCanvas.setAttribute('draggable', 'false');
let auxDrawer = smallCanvas.getContext('2d');

const historySize = 10;
let actionPtr = 0;
let mainActionStack = Array(historySize).fill(null);
mainActionStack[0] = drawer.getImageData(0, 0, mainSize, mainSize);

let auxActionStack = Array(historySize).fill(null);
auxActionStack[0] = auxDrawer.getImageData(0, 0, auxSize, auxSize);

colors = ['red', 'green', 'blue', 'yellow'];

canvas.style.border = '0px solid darkgray';

canvas.addEventListener('click', toggle, {capture: true});
canvas.addEventListener('mousedown', () => down(false));
canvas.addEventListener('mouseup', up);
canvas.addEventListener('mousemove', () => draw(false));
canvas.addEventListener('mouseout', mouseout);
canvas.addEventListener('mouseover', mouseover);

canvas.addEventListener('touchstart', () => down(true));
canvas.addEventListener('touchend', up);
canvas.addEventListener('touchmove', () => draw(true));

body.addEventListener('click', deactivate);

select.addEventListener('click', () => change_tool(select), {capture: true});
pencil.addEventListener('click', () => change_tool(pencil), {capture: true});
eraser.addEventListener('click', () => change_tool(eraser), {capture: true});
trash.addEventListener('click', trashcan, {capture: true});
undo.addEventListener('click', f_undo, {capture: true});
redo.addEventListener('click', f_redo, {capture: true});
submit.addEventListener('click', guess);

function f_undo() {
    event.stopPropagation();
    if (actionPtr-1 >= 0 && mainActionStack[(actionPtr-1)] !== null) {
        actionPtr--;
        drawer.putImageData(mainActionStack[actionPtr], 0, 0);
        auxDrawer.putImageData(auxActionStack[actionPtr], 0, 0);
    }
}

function f_redo() {
    event.stopPropagation();
    if (mainActionStack[(actionPtr+1)] !== null && actionPtr+1 <= mainActionStack.length) {
        actionPtr++;
        auxDrawer.putImageData(auxActionStack[actionPtr], 0, 0);
    }
}

function guess() {
    loading.classList.toggle('collapsed');
    let data = new Blob([auxDrawer.getImageData(0, 0, auxSize, auxSize).data.buffer], {type: 'application/octet-stream'});
    let formData = new FormData();
    formData.append('img', data);
    let xhr = new XMLHttpRequest();
    xhr.timeout = 10000;
    xhr.onload = function () {
        if (xhr.status === 200) {
            document.getElementById('response').placeholder = xhr.responseText
        } else {
            alert('An error occurred!');
        }
        loading.classList.toggle('collapsed');
    };
    xhr.onerror = function () {
        alert('An error occurred!');
        loading.classList.toggle('collapsed');
    };

    xhr.open('POST', '/upload/', true);
    xhr.setRequestHeader('enctype', 'multipart/form-data');
    xhr.send(formData);
}


function trashcan() {
    event.stopPropagation();
    drawer.clearRect(0, 0, mainSize, mainSize);
    auxDrawer.clearRect(0, 0, mainSize, mainSize);
    toggle(canvas);
    save_data();
}


function toggle() {
    event.stopPropagation();
    if (active === false) {
        canvas.classList.replace('select', tool);
        collapser.classList.toggle('collapsed');
        $("#"+wrapper.id).animate({borderWidth: '4px'}, 'fast');
        active = !active;
        console.log('activated')
    }
}

function deactivate() {
    if (active === true) {
        canvas.classList.replace(tool, 'select');
        active = !active;
        collapser.classList.toggle('collapsed');
        $("#"+wrapper.id).animate({borderWidth: '0px'}, 'fast');
        console.log('deactivated')
    }
}

function change_tool(object) {
    event.stopPropagation();
    $("#"+object.id).button('toggle');
    if (active === false) {
        toggle(canvas)
    }
    if (tool !== object.id) {
        canvas.classList.replace(tool, object.id);
        tool = object.id;
        console.log(tool)
    }
}

function down(touch_mode) {
    if (active === true && (tool === 'pencil' || tool === 'eraser')) {
        length = 0;
        drawing = true;
        if (touch_mode === true) {
            prev_x = event.changedTouches[0].clientX - canvas.getBoundingClientRect().left;
            prev_y = event.changedTouches[0].clientY - canvas.getBoundingClientRect().top;
            event.preventDefault();
        } else {
            prev_x = event.clientX - canvas.getBoundingClientRect().left;
            prev_y = event.clientY - canvas.getBoundingClientRect().top;
        }
    }
}

function up(){
    if (drawing === true) {
        collapser.classList.toggle('collapsed');
        if (length > 1) {
            save_data();
        }
        length = 0;
        drawing = false;
    }
}

function mouseover() {
     if (active) {
         prev_x = event.clientX - canvas.getBoundingClientRect().left;
         prev_y = event.clientY - canvas.getBoundingClientRect().top;
     }
}

function mouseout() {
    if (tool === 'pencil' && drawing === true) {
        drawer.stroke();
    }
}

function draw(touch_mode) {

    if (active) {
        let x;
        let y;
        if (drawing) {

            if (collapser.classList.contains('collapsed') === false) {
                collapser.classList.toggle('collapsed');
            }

            length++;
            if (touch_mode === true) {
                x = event.changedTouches[0].clientX - canvas.getBoundingClientRect().left;
                y = event.changedTouches[0].clientY - canvas.getBoundingClientRect().top;
                event.preventDefault();
            } else {
                x = event.clientX - canvas.getBoundingClientRect().left;
                y = event.clientY - canvas.getBoundingClientRect().top;
            }

            if (tool === 'pencil') {
                drawer.lineWidth = mainPencil;
                drawer.globalCompositeOperation = 'source-over';
                auxDrawer.lineWidth = auxPencil;
                auxDrawer.globalCompositeOperation = 'source-over';
            } else {
                drawer.lineWidth = mainEraser;
                drawer.globalCompositeOperation = 'destination-out';
                auxDrawer.lineWidth = auxEraser;
                auxDrawer.globalCompositeOperation = 'destination-out';
            }

            drawer.strokeStyle = 'black';
            drawer.lineCap = 'round';
            drawer.beginPath();
            drawer.moveTo(prev_x, prev_y);
            drawer.lineTo(x, y);
            drawer.stroke();

            auxDrawer.strokeStyle = 'black';
            auxDrawer.lineCap = 'round';
            auxDrawer.beginPath();
            auxDrawer.moveTo(prev_x * scalingFactor, prev_y * scalingFactor);
            auxDrawer.lineTo(x * scalingFactor, y * scalingFactor);
            auxDrawer.stroke();

            prev_x = x;
            prev_y = y;
            console.log(x + ' ' + y)
        }
    }
}

function dist(x1,y1,x2,y2) {
	x2-=x1;
	y2-=y1;
	return Math.sqrt((x2*x2) + (y2*y2));
}

function save_data() {
    if (mainActionStack[actionPtr+1] !== null) {
        mainActionStack.fill(null, actionPtr+1);
        auxActionStack.fill(null, actionPtr+1);
    }
    if (actionPtr+1 >= mainActionStack.length) {
        let mainTemp = mainActionStack.slice(historySize/2, historySize);
        let auxTemp = auxActionStack.slice(historySize/2, historySize);

        mainActionStack.fill(null);
        auxActionStack.fill(null);

        mainActionStack = mainTemp.concat(Array(historySize/2).fill(null));
        auxActionStack = auxTemp.concat(Array(historySize/2).fill(null));

        actionPtr = historySize/2;
        mainActionStack[actionPtr] = drawer.getImageData(0, 0, mainSize, mainSize);
        auxActionStack[actionPtr] = auxDrawer.getImageData(0, 0, mainSize, mainSize);
    } else {
        mainActionStack[(actionPtr + 1)] = drawer.getImageData(0, 0, mainSize, mainSize);
        auxActionStack[(actionPtr + 1)] = auxDrawer.getImageData(0, 0, mainSize, mainSize);
        actionPtr++;
    }
}

