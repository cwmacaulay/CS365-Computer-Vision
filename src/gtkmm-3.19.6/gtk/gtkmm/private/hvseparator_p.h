// Generated by gmmproc 2.47.4 -- DO NOT MODIFY!
#ifndef _GTKMM_HVSEPARATOR_P_H
#define _GTKMM_HVSEPARATOR_P_H
#ifndef GTKMM_DISABLE_DEPRECATED


#include <gtkmm/private/separator_p.h>

#include <glibmm/class.h>

namespace Gtk
{

class VSeparator_Class : public Glib::Class
{
public:
#ifndef DOXYGEN_SHOULD_SKIP_THIS
  typedef VSeparator CppObjectType;
  typedef GtkVSeparator BaseObjectType;
  typedef GtkVSeparatorClass BaseClassType;
  typedef Gtk::Separator_Class CppClassParent;
  typedef GtkSeparatorClass BaseClassParent;

  friend class VSeparator;
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

  const Glib::Class& init();


  static void class_init_function(void* g_class, void* class_data);

  static Glib::ObjectBase* wrap_new(GObject*);

protected:

  //Callbacks (default signal handlers):
  //These will call the *_impl member methods, which will then call the existing default signal callbacks, if any.
  //You could prevent the original default signal handlers being called by overriding the *_impl method.

  //Callbacks (virtual functions):
};


} // namespace Gtk


#include <glibmm/class.h>

namespace Gtk
{

class HSeparator_Class : public Glib::Class
{
public:
#ifndef DOXYGEN_SHOULD_SKIP_THIS
  typedef HSeparator CppObjectType;
  typedef GtkHSeparator BaseObjectType;
  typedef GtkHSeparatorClass BaseClassType;
  typedef Gtk::Separator_Class CppClassParent;
  typedef GtkSeparatorClass BaseClassParent;

  friend class HSeparator;
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

  const Glib::Class& init();


  static void class_init_function(void* g_class, void* class_data);

  static Glib::ObjectBase* wrap_new(GObject*);

protected:

  //Callbacks (default signal handlers):
  //These will call the *_impl member methods, which will then call the existing default signal callbacks, if any.
  //You could prevent the original default signal handlers being called by overriding the *_impl method.

  //Callbacks (virtual functions):
};


} // namespace Gtk

#endif // GTKMM_DISABLE_DEPRECATED


#endif /* _GTKMM_HVSEPARATOR_P_H */

