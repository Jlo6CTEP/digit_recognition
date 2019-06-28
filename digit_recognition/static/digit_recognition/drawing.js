var toolkit = ['select', 'eraser', 'pencil'];

var wrapper = document.getElementById('wrapper');
var canvas = document.getElementById('drawing_board');
var eraser = document.getElementById('eraser');
var pencil = document.getElementById('pencil');
var select = document.getElementById('select');
var trash = document.getElementById('trash');
var collapser = document.getElementById('collapser');

var drawer = canvas.getContext('2d');
var body = document.body;

var active = false;
var drawing = false;
var tool = 'select';

var prev_x = 0;
var prev_y = 0;

const square_clear = 30;


colors = ['red', 'green', 'blue', 'yellow'];

drawer.fillStyle = 'red';
canvas.style.border = '0px solid darkgray';

canvas.addEventListener('click', toggle, {capture: true});
canvas.addEventListener('mousedown', mousedown);
canvas.addEventListener('mouseup', mouseup);
canvas.addEventListener('mousemove', draw);
canvas.addEventListener('mouseout', mouseout);
canvas.addEventListener('mouseover', mouseover);
body.addEventListener('click', deactivate);

select.addEventListener('click', () => change_tool(select), {capture: true});
pencil.addEventListener('click', () => change_tool(pencil), {capture: true});
eraser.addEventListener('click', () => change_tool(eraser), {capture: true});
trash.addEventListener('click', trashcan, {capture: true});

function trashcan() {
    event.stopPropagation();
    drawer.clearRect(0, 0, 477, 477);
    toggle(canvas)
}


function toggle() {
    event.stopPropagation();
    if (active === false) {
        canvas.classList.replace('select', tool);
        collapser.classList.toggle('collapsed');
        $("#"+wrapper.id).animate({borderWidth: '4px'}, 'fast');
        active = !active;
        console.log('activated')
    } else {

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

function mousedown() {
    if (active === true && (tool === 'pencil' || tool === 'eraser')) {
        collapser.classList.toggle('collapsed');
        drawing = true;
        prev_x = event.clientX - canvas.getBoundingClientRect().left;
        prev_y = event.clientY - canvas.getBoundingClientRect().top;
    }
}

function mouseup() {
    if (drawing === true) {
        collapser.classList.toggle('collapsed');
    }
    drawing = false;
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

function draw() {

    if (active) {
        let x;
        let y;
        if (drawing) {
            x = event.clientX - canvas.getBoundingClientRect().left;
            y = event.clientY - canvas.getBoundingClientRect().top;

            if (tool === 'pencil') {
                drawer.strokeStyle = 'black';
                drawer.lineCap = 'round';

                drawer.beginPath();
                drawer.moveTo(prev_x, prev_y);
                drawer.lineTo(x, y);
                drawer.lineWidth = 5;
                drawer.stroke();
            } else {
                let tmp, length;

                let x1 = prev_x, x2 = x, y1 = prev_y, y2 = y;

                // swap coordinate pairs if x-coordinates are RTL to make them LTR
                if (x2 < x1) {
                    tmp = x1;
                    x1 = x2;
                    x2 = tmp;
                    tmp = y1;
                    y1 = y2;
                    y2 = tmp;
                }

                length = dist(x1, y1, x2, y2) + 5;

                drawer.save();
                drawer.translate(x1, y1);
                drawer.rotate(Math.atan2(y2 - y1, x2 - x1));
                drawer.clearRect(0, -7.5, length, 15);
                drawer.restore();
            }
            prev_x = x;
            prev_y = y;
            //console.log(x + ' ' + y)
        }
    }
}

function dist(x1,y1,x2,y2) {
	x2-=x1; y2-=y1;
	return Math.sqrt((x2*x2) + (y2*y2));
}

