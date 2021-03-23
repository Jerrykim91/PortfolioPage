


var triggerTabList = [].slice.call(document.querySelectorAll('#myList a'))
triggerTabList.forEach(function (triggerEl) {
  var tabTrigger = new bootstrap.Tab(triggerEl)

  triggerEl.addEventListener('click', function (event) {
    event.preventDefault()
    tabTrigger.show()
  })
})


var someListItemEl = document.querySelector('#someListItem')
  var tab = new bootstrap.Tab(someListItemEl)

  tab.show()