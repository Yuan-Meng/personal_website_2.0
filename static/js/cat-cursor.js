// Add the cat-cursor class
function addCatCursor() {
  document.body.classList.add("cat-cursor");
}

// Remove the cat-cursor class with a delay for slower reverting
function removeCatCursorWithDelay(delay = 300) { // Default delay of 300ms
  setTimeout(() => {
    document.body.classList.remove("cat-cursor");
  }, delay);
}

// Handle mouse click
document.addEventListener("mousedown", addCatCursor); // Add the cat cursor on mouse down
document.addEventListener("mouseup", () => removeCatCursorWithDelay(300)); // Remove the cat cursor more slowly (500ms delay)

// Handle scrolling
let isScrolling;
window.addEventListener("scroll", () => {
  addCatCursor(); // Add the cat cursor during scrolling

  // Clear any previously set timeout
  clearTimeout(isScrolling);

  // Set a timeout to remove the cat cursor more slowly after scrolling stops
  isScrolling = setTimeout(() => removeCatCursorWithDelay(300), 100);
});
