function setupDynamicStocks(containerId, addBtnId, minCount, withWeights) {
  const container = document.getElementById(containerId);
  const addBtn = document.getElementById(addBtnId);
  if (!container || !addBtn) return;

  // grab template options
  const stockOptions = document.getElementById("stock-options-template").innerHTML;

  addBtn.addEventListener("click", () => {
    const div = document.createElement("div");
    div.className = "stock-input-group mb-2";
    div.innerHTML = `
      <select name="stock" class="form-control">
        <option value="">Select a Stock</option>
        ${stockOptions}
      </select>
      ${withWeights ? '<input type="number" name="weight" placeholder="Amount (%)" step="0.01" class="form-control">' : ""}
      <button type="button" class="btn btn-danger remove-stock-btn">Remove</button>`;
    container.appendChild(div);
  });

  container.addEventListener("click", (e) => {
    if (e.target.classList.contains("remove-stock-btn")) {
      if (container.querySelectorAll(".stock-input-group").length > minCount) {
        e.target.closest(".stock-input-group").remove();
      } else {
        alert(`You must have at least ${minCount} stocks.`);
      }
    }
  });
}

document.addEventListener("DOMContentLoaded", () => {
  setupDynamicStocks("stocks-container", "add-stock-btn", 2, true);
  setupDynamicStocks("stocks-risk-container", "add-risk-stock-btn", 2, false);
  setupDynamicStocks("stocks-return-container", "add-return-stock-btn", 2, false);
});
