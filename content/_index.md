---
title: "Probably Approximately Correct"
---

An ML engineer's musings on machine learning and life...
{.lead}

<div class="sudoku-container" id="container"></div>

<div id="message" class="game-message">In the mood for a Sudoku challenge? 🍀</div>

<div class="buttonContainer">
    <button id="solveButton" class="styled-button">Solve</button>
    <button id="resetButton" class="styled-button">Reset</button>
</div>

<script>
document.addEventListener("DOMContentLoaded", function () {
    const container = document.getElementById("container");
    const messageElement = document.getElementById("message");

    let maskedPuzzle;
    let correctPuzzle;
    let remainingCells;

    function generateRandomSudoku() {
        const puzzle = [
            [5, 3, 0, 0, 7, 0, 0, 0, 0],
            [6, 0, 0, 1, 9, 5, 0, 0, 0],
            [0, 9, 8, 0, 0, 0, 0, 6, 0],
            [8, 0, 0, 0, 6, 0, 0, 0, 3],
            [4, 0, 0, 8, 0, 3, 0, 0, 1],
            [7, 0, 0, 0, 2, 0, 0, 0, 6],
            [0, 6, 0, 0, 0, 0, 2, 8, 0],
            [0, 0, 0, 4, 1, 9, 0, 0, 5],
            [0, 0, 0, 0, 8, 0, 0, 7, 9],
        ];
        return puzzle;
    }

    function solveSudoku(board) {
        const solvedBoard = JSON.parse(JSON.stringify(board));
        solveHelper(solvedBoard);
        return solvedBoard;
    }

    function solveHelper(board) {
        const emptyCell = findEmptyCell(board);
        if (!emptyCell) return true;

        const [row, col] = emptyCell;
        for (let num = 1; num <= 9; num++) {
            if (isValidMove(board, row, col, num)) {
                board[row][col] = num;
                if (solveHelper(board)) return true;
                board[row][col] = 0; // Backtrack
            }
        }
        return false;
    }

    function findEmptyCell(board) {
        for (let row = 0; row < 9; row++) {
            for (let col = 0; col < 9; col++) {
                if (board[row][col] === 0) return [row, col];
            }
        }
        return null;
    }

    function isValidMove(board, row, col, num) {
        for (let i = 0; i < 9; i++) {
            if (board[row][i] === num || board[i][col] === num) return false;
        }
        const startRow = Math.floor(row / 3) * 3;
        const startCol = Math.floor(col / 3) * 3;
        for (let i = startRow; i < startRow + 3; i++) {
            for (let j = startCol; j < startCol + 3; j++) {
                if (board[i][j] === num) return false;
            }
        }
        return true;
    }

    function createSudokuGrid(puzzle) {
        container.innerHTML = '';
        remainingCells = 0;

        puzzle.forEach((row, rowIndex) => {
            const rowElement = document.createElement('div');
            rowElement.classList.add('sudoku-row');

            row.forEach((cell, columnIndex) => {
                const cellElement = document.createElement('input');
                cellElement.classList.add('sudoku-cell');
                cellElement.type = 'text';
                cellElement.maxLength = 1;

                if (cell !== 0) {
                    cellElement.value = cell;
                    cellElement.disabled = true;
                    cellElement.style.backgroundColor = '#FDB515';
                } else {
                    cellElement.style.backgroundColor = '#002676';
                    cellElement.style.color = 'white';
                    remainingCells++;
                    cellElement.addEventListener('input', (event) => handleInput(event, rowIndex, columnIndex));
                }

                // Add 3x3 box borders
                if (rowIndex % 3 === 0) cellElement.style.borderTop = '3px solid black';
                if (columnIndex % 3 === 0) cellElement.style.borderLeft = '3px solid black';
                if (rowIndex === 8) cellElement.style.borderBottom = '3px solid black';
                if (columnIndex === 8) cellElement.style.borderRight = '3px solid black';

                rowElement.appendChild(cellElement);
            });
            container.appendChild(rowElement);
        });
    }

    function handleInput(event, row, col) {
        const input = event.target.value;
        const num = parseInt(input, 10);

        if (isNaN(num) || num < 1 || num > 9) {
            event.target.value = '';
            messageElement.textContent = "Please choose a number from 1 to 9! 🥶";
            messageElement.style.color = "red";
            return;
        }

        if (!isValidMove(maskedPuzzle, row, col, num)) {
            messageElement.textContent = "Oops, please pick another number! 😬";
            messageElement.style.color = "red";
        } else {
            messageElement.textContent = "Keep working! 💪";
            messageElement.style.color = "inherit";
            maskedPuzzle[row][col] = num;
            remainingCells--;

            if (remainingCells === 0) {
                messageElement.textContent = "Amazing! You crushed it! 🎉";
                messageElement.style.color = "green";
            }
        }
    }

    function solvePuzzle() {
        createSudokuGrid(correctPuzzle);
        messageElement.textContent = "Here's the solution! 🧩";
        messageElement.style.color = "green";
    }

    function resetPuzzle() {
        maskedPuzzle = JSON.parse(JSON.stringify(correctPuzzle));
        maskedPuzzle.forEach((row, rowIndex) => {
            row.forEach((cell, colIndex) => {
                if (Math.random() < 0.5) maskedPuzzle[rowIndex][colIndex] = 0;
            });
        });
        createSudokuGrid(maskedPuzzle);
        messageElement.textContent = "In the mood for a Sudoku challenge? 🍀";
        messageElement.style.color = "inherit";
    }

    correctPuzzle = generateRandomSudoku();
    solveHelper(correctPuzzle);

    resetPuzzle();

    document.getElementById("solveButton").addEventListener("click", solvePuzzle);
    document.getElementById("resetButton").addEventListener("click", resetPuzzle);
});
</script>

<style>
.sudoku-container {
    display: flex;
    flex-direction: column;
    align-items: center;
    margin-top: 20px;
}

.sudoku-row {
    display: flex;
}

.sudoku-cell {
    width: 30px;
    height: 30px;
    text-align: center;
    font-size: 16px;
    box-sizing: border-box;
}

.buttonContainer {
    margin-top: 20px;
    display: flex;
    justify-content: center;
    gap: 15px;
}

.styled-button {
    padding: 5px 10px;
    font-size: 14px;
    color: #333;
    background-color: #f9f9f9;
    border: 2px solid #ccc;
    border-radius: 4px;
    cursor: pointer;
}

.styled-button:hover {
    background-color: #ddd;
}

.game-message {
    text-align: center;
    margin-top: 15px;
    font-size: 14px;
}
</style>
